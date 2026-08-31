# Design Report: End-to-End Occupancy Perception

## 1. Approach to generating occupancy ground truth

The goal of this stage is to turn raw nuScenes sensor data into a voxel grid occupancy target for a learning pipeline. The grid is fixed at 200 by 200 by 16 voxels covering an 80 metre by 80 metre region around the ego vehicle at 0.4 metre resolution, following the Occ3D-nuScenes convention. It is defined in the ego frame, so all sensor data is transformed into that frame before voxelization.

The pipeline runs per keyframe in five stages.

The first stage is multi-sweep aggregation. A single sweep from the 32-beam LiDAR is sparse, leaving many truly occupied voxels empty. Ten consecutive sweeps are aggregated to make the cloud denser. Because the ego vehicle moves between sweeps, the sweeps cannot simply be concatenated. Each sweep's points are transformed through the sensor to ego to global chain into a common frame before merging. This gives roughly a tenfold increase in density, from about 26,000 points per sweep to about 263,000 aggregated points.

The second stage is a frame correction. The aggregated cloud is expressed in the LiDAR sensor frame, which sits about 1.8 metres above the ego origin on the vehicle roof. Before voxelization the points are transformed into the ego frame using the calibrated sensor extrinsic. Skipping this would shift all occupancy upward by several voxels.

The third stage is semantic labeling. Each point is labeled by the 3D bounding box that contains it, giving the foreground classes such as car, truck, pedestrian, and barrier. Points that fall inside no box remain unlabeled. In practice these are the background, meaning road, vegetation, and buildings, which the boxes do not describe.

The fourth stage is visibility through ray casting. This is the stage that separates real occupancy ground truth from a voxelized point dump. A voxel with no points is ambiguous. It may be genuinely free, meaning a beam passed through it, or it may be unobserved, meaning it was occluded and no beam reached it. Labeling occluded space as free would train a model to imagine free space behind obstacles. The pipeline resolves three states, which are occupied, free, and unobserved, by casting a ray from the sensor origin to each hit point using an exact voxel traversal algorithm from Amanatides and Woo. Voxels that a ray passes through are marked free, the endpoint voxel is marked occupied, and voxels that no ray reaches stay unobserved.

The fifth stage is serialization. Each keyframe's label grid and visibility grid are written to a compressed npz file. This separates the expensive generation stage from training, so the learning pipeline reads ready-made arrays with no dependency on the nuScenes devkit.

The two bird's-eye-view figures in the README show this correspondence directly. Where the aggregated LiDAR has returns, the occupancy grid marks occupied or free. Where the LiDAR is empty, in particular the radial wedges behind obstacles, the grid marks unobserved. These occlusion shadows are visible in both views and confirm that the ground truth reflects what the sensor actually observed, not merely where points happened to land.

## 2. Key deep learning pipeline modules and how they interact

The learning pipeline is deliberately modular. Each stage lives in its own module, and a config object describes a run.

The dataset module is a torch Dataset that reads the serialized npz files. For each sample it returns a single channel binary occupancy input, a semantic target with unobserved voxels set to an ignore value, and an observed mask. It imports only the serialization module and never the generation pipeline, so training has no dependency on the nuScenes devkit. The number of classes is read from the stored category map, so the model and the data stay in step automatically.

The model is a placeholder network assembled as an encoder, a neck, and a head, each pulled from a registry by name. The encoder is two 3D convolution stages, and the second stage halves the resolution. The neck is a bottleneck. The head upsamples back to full resolution and predicts a class score per voxel. The registry means any block can be replaced by a one line config change, for example swapping the simple encoder for a 3D ResNet, with no edit to the assembly code. The output aligns voxel for voxel with the target grid.

The loss is a masked cross-entropy that supervises only observed voxels. Unobserved voxels carry an ignore value of minus one, so the model is never trained on space the sensor could not see. A batch with no observed voxels returns a differentiable zero rather than a NaN, so a degenerate batch cannot corrupt training.

The trainer is a plain PyTorch loop. It builds the components from a config, then runs the forward pass, the masked loss, the backward pass, and the optimizer step, logging the loss at each step. Its purpose is to show that the pipeline is training ready and end to end differentiable, not to train a model to convergence.

The config is a dataclass holding every setting, meaning paths, model shape, and optimization values, and it can be overridden from a YAML file. It validates its values at construction so a bad setting fails immediately. The device is detected automatically, so the same config runs on a GPU or a CPU without change.

The data flow is straightforward. Serialized ground truth is read by the dataset, batched by the data loader, passed through the model to produce logits, compared to the target by the masked loss, and used to update the weights through backpropagation. The class count flows from the dataset into both the model head and the loss, so the two cannot fall out of step.

## 3. Design trade-offs and assumptions

The model is a placeholder. The task grades pipeline design rather than model quality and states that a placeholder is sufficient. Effort was therefore directed at the ground truth logic and the pipeline structure rather than a trained model. The encoder, neck, and head design together with the registry make a real backbone a drop-in replacement, which is treated as an extension.

Dynamic objects lag slightly. Labels come from the keyframe's 3D boxes, but points come from ten sweeps spanning up to about half a second of motion. Points from older sweeps are labeled at their recorded positions, so points on moving objects lag their keyframe time boxes a little. This is the standard trade-off for multi-sweep occupancy ground truth. Setting the sweep count to one avoids it at the cost of density. Aggregating each object in its own local frame would remove it and is noted as an extension.

The visibility rays share one origin. All visibility rays are cast from the keyframe sensor position, but the true origins of older sweeps were metres away. Free space along those rays is therefore slightly biased near fast ego motion. Casting each sweep from its own origin would remove this and is noted as an extension.

The background is unlabeled. The 3D boxes cover only foreground classes, so most of the occupied space, meaning road, vegetation, and buildings, stays unlabeled. Adding nuScenes-lidarseg for per-point background labels is the natural refinement and is left as a plug-in point.

Storage is dense. The grids are stored dense and compressed rather than sparse. Occupancy is only a few percent of the voxels, so compression removes most of the redundancy, and each keyframe compresses to well under 100 kilobytes. The training pipeline then receives ready-to-use dense arrays. A sparse format is noted as an optimization but is not needed at this scale.

The occupancy threshold favours completeness. A voxel is occupied if it contains at least one aggregated point. Completeness is chosen over noise filtering, because a missed obstacle is a worse error than a spurious one for navigation, and multi-sweep aggregation already suppresses most stray returns. The threshold is exposed as a configurable value.

## 4. Potential improvements and extensions

Several extensions follow naturally from the design.

The first is adding nuScenes-lidarseg for background labels, which would give semantic classes for road, vegetation, and structures that the boxes do not cover. The labeling module is structured so this can slot in as an additional label source.

The second is per-instance aggregation of dynamic objects, transforming each moving object's points into its own box-local frame so that dynamic objects do not smear or lag.

The third is casting visibility rays from per-sweep origins, which would remove the origin bias in free space near fast motion.

The fourth is a real 3D backbone, such as a 3D ResNet or sparse convolutions, swapped in through the registry, together with richer input features such as raw point features or camera bird's-eye-view fusion instead of the single occupancy channel.

The fifth is stronger losses for the heavy class imbalance, such as Lovasz-softmax or geometric and semantic scaled losses. Class weighting is already available in the loss and is off by default.

The sixth is temporal accumulation across a whole scene, which would shrink the unobserved region, since a single viewpoint leaves most of the scene occluded or out of range.

More broadly, an occupancy grid of this kind is a natural scene representation for downstream planning and decision making in an autonomous driving system, so this perception module could feed the reasoning and planning stages of a larger pipeline.

## Acknowledgements and tool usage

This project uses the nuScenes devkit for data loading, multi-sweep aggregation, and coordinate transforms, and PyTorch for the deep learning pipeline. The occupancy ground-truth methodology follows the conventions of Occ3D-nuScenes and OpenOccupancy. The visibility ray casting uses the voxel traversal algorithm from Amanatides and Woo, published in 1987.

A large language model (claude) was used as an assistant during development. It helped with scaffolding module structure, drafting boilerplate and docstrings, suggesting standard techniques, and debugging. The design decisions, the choice and arrangement of the pipeline stages, the geometry and coordinate handling, the ground truth logic, and the final implementation were reasoned through and verified by me. Every module was tested against real nuScenes data to confirm correct behaviour.