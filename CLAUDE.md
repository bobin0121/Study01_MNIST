# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A small PyTorch project that trains a CNN on MNIST and lets a user hand-draw a digit in a Tkinter canvas for live recognition. All code comments and UI text are written in Korean.

## Commands

Python 3.11 is required. On this machine it lives at `C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe` (the bare `python`/`python3` on PATH may resolve to the Microsoft Store stub in some shells — if so, invoke the full path or restart the shell so the installed Python's PATH entry takes effect).

Install dependencies:
```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install pillow numpy
```

Train the model (downloads MNIST into `./data` on first run, trains 5 epochs, saves weights to `mnist_cnn.pt`):
```
python train.py
```

Run the handwriting recognition GUI (requires `mnist_cnn.pt` to already exist):
```
python predict_gui.py
```

There is no test suite, linter, or build step in this project.

## Architecture

- [model.py](model.py) — `MnistCNN`: the only model definition, shared by both `train.py` and `predict_gui.py`. Architecture is fixed at conv(1→32)→pool→conv(32→64)→pool→dropout→fc(3136→128)→dropout→fc(128→10), assuming 28x28 single-channel input. Changing this architecture requires re-running `train.py` to regenerate `mnist_cnn.pt`, since the GUI loads weights via `load_state_dict` and will fail on shape mismatches.
- [train.py](train.py) — Loads MNIST via `torchvision.datasets.MNIST` (normalized with the standard MNIST mean/std `0.1307`/`0.3081`), trains `MnistCNN`, evaluates test accuracy after every epoch, and writes weights to `mnist_cnn.pt` via `torch.save(model.state_dict(), ...)`. Hyperparameters (`BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `WEIGHT_PATH`) are module-level constants at the top of the file.
- [predict_gui.py](predict_gui.py) — Tkinter app (`DigitRecognizerApp`) with a 280x280 black drawing canvas. Mouse drag events are painted both onto the visible Tkinter canvas and in parallel onto an in-memory PIL image (`self.image`), which is the actual data fed to the model. On "인식하기" (predict), the PIL image is downsampled to 28x28, normalized with the same mean/std used in training, and run through `MnistCNN` to produce a predicted digit and confidence. The normalization constants here must stay in sync with `train.py`'s — they are duplicated, not shared.
- `data/` and `__pycache__/` are generated artifacts (downloaded MNIST files and compiled bytecode); do not hand-edit or expect them to be portable.
