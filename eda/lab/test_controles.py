"""Prueba en seco de recommender-v4.controles.py con la descarga en caché del laboratorio (sin tocar RawTree ni la caché real)."""
import os, sys, pickle, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rc", HERE / "recommender-v4.controles.py")
rc = importlib.util.module_from_spec(spec); sys.modules["rc"] = rc; spec.loader.exec_module(rc)
rc.CACHE = HERE / ".cache"; rc.LOGS = HERE / "logs"
dl = pickle.loads((HERE / ".cache" / "dl.pkl").read_bytes()); rc.download_feedback = lambda rt: dl
rc.setup_logging(False); rc.calibrate_live(rc.make_rt(), rc.CFG, args_dry=True)
