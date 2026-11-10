# Third-party components

FlexSfM Studio is licensed under Apache-2.0. It depends on the projects below, which are installed separately
(pip / system packages) and are **not** included in this repository. Check each licence before redistributing a
bundled build or using the tool commercially.

| Component | Used for | Licence |
|---|---|---|
| [FlexGlue](https://github.com/cvg/FlexGlue) | feature matching, dense descriptors | MIT |
| DINOv3 weights (downloaded by FlexGlue) | FlexGlue's backbone | [Meta DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license) - custom terms, review before commercial use |
| [LightGlue](https://github.com/cvg/LightGlue) (ALIKED detector) | optional ALIKED keypoints | Apache-2.0 (ALIKED: BSD-3-Clause) |
| [COLMAP / GLOMAP](https://github.com/colmap/colmap) via pycolmap | reconstruction, bundle adjustment, undistortion | BSD-3-Clause |
| [PyTorch](https://pytorch.org) | running FlexGlue | BSD-3-Clause |
| [OpenCV](https://opencv.org) | SIFT, RANSAC/MAGSAC, image I/O | Apache-2.0 |
| NumPy, SciPy, Pillow, PyOpenGL | numerics, image I/O, 3D view | BSD-style |
| Matplotlib | figures | Matplotlib licence (PSF-style) |
| PyYAML, pyqtgraph | config, 3D/plots in the app | MIT |
| [PySide6 / Qt](https://www.qt.io/qt-for-python) | desktop app | LGPL-3.0 (used as a dynamically linked library) |
| [FFmpeg](https://ffmpeg.org) (system binary) | frame extraction from video | LGPL-2.1+ / GPL depending on build |

## Demo data

The screenshots and `scripts/get_demo_data.sh` use the OpenDroneMap **aukerman** dataset
([odm_data_aukerman](https://github.com/OpenDroneMap/odm_data_aukerman)), released into the public domain under CC0 1.0.
The dataset is downloaded on demand and is not part of this repository.
