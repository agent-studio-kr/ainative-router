"""RouterArena 스크립트를 우리 venv + 사내 TLS 설정으로 실행: python ra_run.py <script> [args...]"""
import runpy, sys
sys.path.insert(0, "/Users/kdb/Desktop/ai-model-router/scripts"); import _tls  # noqa
sys.path.insert(0, "/Users/kdb/Desktop/ai-model-router/third_party/RouterArena")
import os
script = sys.argv[1]; sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))  # python <script> 와 동일하게
runpy.run_path(script, run_name="__main__")
