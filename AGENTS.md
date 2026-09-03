# AGENTS.md — Development Conventions & Engineering Guidelines

This document outlines the architecture, coding standards, and operational guidelines for AI agents working in this repository.

---

## 1. Environment & Python Runtime
* **Python Version**: Strictly Python 3.11 (`.venv\Scripts\python.exe`). Do NOT invoke Python 3.14.
* **Virtual Environment**: All commands, package installations, and script runs MUST use `.venv\Scripts\...`.
* **Hardware Profile**:
  * GPU: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA 12.x).
  * CPU: 12th Gen Intel Core i5-12450HX (8 cores, 12 threads).
  * System Memory: 16 GB RAM.

---

## 2. Core Architectural Principles
1. **Never Make Definitive Medical Diagnoses**:
   * The system is a decision-support and safety alert tool.
   * Notifications MUST ALWAYS say: *"Possible medical emergency / abnormal behavior detected. Please check the person."*
   * Never state: *"Stroke confirmed"* or *"Patient fractured hip"*.
2. **Temporal Reasoning Over Frame-Level Snapshots**:
   * A person lying on the floor alone is not necessarily an emergency (they may be stretching or resting).
   * Emergency requires temporal confirmation: `standing -> sudden drop (high Ay) -> impact -> prolonged immobility`.
3. **Low False Alarm Rate & Debouncing**:
   * Any change in state must pass through the debounced state machine (`src/risk/state_machine.py`).
   * Quick bends (< 2s) to pick up items must not trigger fall alarms.
4. **VRAM Budget Discipline**:
   * Do NOT load redundant heavy 3D-CNNs concurrently.
   * The pipeline relies on: `YOLOv8s-Pose` + `ByteTrack` + `Spatial-Temporal Transformer` on normalized skeletons.

---

## 3. Code Quality & Standards
* **Typing**: Use standard Python type annotations everywhere (`list[float]`, `dict[str, Any]`, `tuple[int, ...]`, `Optional[...]`).
* **Interfaces**: Every module must inherit from its corresponding Abstract Base Class (`CameraStream`, `PersonDetector`, `Tracker`, `PoseEstimator`, `AnomalyDetector`).
* **Configuration**: Never hard-code thresholds, camera URLs, or model weights. Load from `configs/*.yaml`.
* **Testing**: All new features must have unit tests in `tests/`.
