"""Loads the frozen export (models, config, replay pack) and checks its sha256 manifest."""
import hashlib, json, sys
from pathlib import Path


class Export:
    def __init__(self, root, verify=True):
        self.root = Path(root); self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if verify: self.verify()
        code = str(self.root / 'code')
        if code not in sys.path: sys.path.insert(0, code)
        import b2_lib, c03_infer, c03_export                                   # the frozen libraries shipped inside the export
        self.B, self.c, self.X = b2_lib, c03_infer, c03_export
        self.config = json.loads((self.root / 'config' / 'engine_config.json').read_text())
        self.LR, self.gen_stats, self.fusion_cfg = c03_export.load_fusion(self.root)
        self.symbols = self.config['symbols']; cd = self.root / 'config'
        self.decl = {k: json.loads((cd / f).read_text()) for k, f in {'final': 'declared_final_set_v1.json', 'ae': 'declared_ae_v2.json', 'fusion': 'declared_fusion_v0.json', 'watch': 'declared_watch_v1.json', 'mh': 'declared_multihorizon_v2.json', 'unc': 'declared_uncertainty_v1.json'}.items()}
        M = c03_infer.MODEL_NAMES; self.model_q = {M[0]: self.decl['ae']['q_T1_by_budget']['2'], M[1]: self.decl['final']['models'][M[1]]['q_T1_by_budget']['2'], M[2]: self.decl['final']['models'][M[2]]['q_T1_by_budget']['2']}

    def verify(self):
        bad = []
        for rel, meta in self.manifest['files'].items():
            p = self.root / rel
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != meta['sha256']: bad.append(rel)
        if bad: raise RuntimeError('export files changed or missing (sha256 mismatch): ' + ', '.join(bad[:5]))

    def bundle(self, sym, device='cpu'): return self.X.load_bundle_from_export(self.root, sym, device)
