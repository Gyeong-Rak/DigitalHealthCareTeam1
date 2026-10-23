"""NVIDIA apex 대체 모듈 (서버에서 apex CUDA 빌드 불가).

EVA-X는 apex를 import만 하고 기본 설정에서는 PyTorch 기본 연산과 동일하게 동작하므로
PyTorch 구현으로 대체한다. scripts/run_chexpert.sh가 PYTHONPATH에 compat/를 추가한다.
"""
