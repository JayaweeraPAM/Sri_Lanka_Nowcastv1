# Ceylon Skies: AI Weather Nowcasting for Sri Lanka

Ceylon Skies is a deep learning-based weather nowcasting system designed to predict short-term cloud movements and intensity variations specifically over the Sri Lankan region. By shifting away from traditional computational physics models, this project utilizes a **Convolutional Long Short-Term Memory (ConvLSTM)** network to treat high-resolution satellite imagery sequences as a numerical density field prediction problem. 

The pipeline processes raw meteorological data, crops it to the Sri Lankan coordinate bounding box, handles spatial interpolation/regridding, and prepares continuous time-series arrays optimized for high-performance GPU training loops.

---

## 📂 Project Structure

The repository follows a clean, modular architecture designed for scalability, separating data engineering workflows from deep learning training and model evaluation.

```text
Sri_Lanka_Nowcast/
├── configs/           # Configuration files (.yaml or .json) for hyperparams
├── data/              # Dataset storage directory
│   ├── raw/           # Original downloaded EUMETSAT GRIB files (.grb)
│   ├── processed/     # Spatial regridded and normalized frames (.npy)
│   └── checked_temporal_time_gaps_after_preprocess/  # Verified continuous sequences
├── src/               # Main source code
│   ├── data/          # Data loading pipelines, custom PyTorch Datasets
│   ├── models/        # ConvLSTM model architectures
│   └── utils/         # Metrics, logging, and visualization helpers
├── scripts/           # Standalone verification and utility tools
├── checkpoints/       # Saved PyTorch model weights (.pt / .pth)
├── outputs/           # Rendered model previews and visualization maps
├── README.md          # Project documentation
└── .gitignore         # Rules to exclude heavy data/weights from Git tracking
