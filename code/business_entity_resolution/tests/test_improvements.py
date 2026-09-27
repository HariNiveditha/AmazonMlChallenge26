"""Unit tests for pipeline improvements (synthetic, no dataset needed).

Covers: French open-set normalisation, prior-correction math, tree-model
fit/predict/save-load roundtrip, threshold grid, and baseline defaults.
Run: python -m unittest discover -s tests -v  (from business_entity_resolution/)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from business_entity_resolution.src.config import PipelineConfig
from business_entity_resolution.src.features import FEATURE_NAMES, extract_features
from business_entity_resolution.src.model import (
    MatchingModel,
    macro_f05,
    prior_correct_proba,
)
from business_entity_resolution.src.normalize import (
    normalize_address,
    normalize_name,
)

PAIRS = [
    ("Acme Corporation", "123 Main Street, Springfield, IL 62701",
     "Acme Corp", "123 Main St, Springfield, IL 62701", 1),
    ("Boulangerie Saint Honore SAS", "24 Rue Saint Honore, 75001 Paris",
     "Boulangerie St Honore", "24 Rue St Honore, Paris 75001", 1),
    ("Sri Balaji Traders", "12 MG Road, Bengaluru 560001",
     "Sri Balaji Enterprises", "12 MG Road, Bengaluru 560001", 0),
    ("Blue Ribbon Bakery", "45 Baker Road, Austin, TX 78701",
     "Blue Ribbon Bakery", "45 Baker Road, Austin, TX 78701", 1),
    ("Tech Solutions Inc", "500 Innovation Drive, San Jose, CA",
     "Ferme Auberge du Sud", "Route de Marseille, 13200 Arles", 0),
]


def make_xy():
    rows, labels = [], []
    for n1, a1, n2, a2, y in PAIRS:
        rows.append(extract_features(
            normalize_name(n1), normalize_address(a1), "US",
            normalize_name(n2), normalize_address(a2), "US", None))
        labels.append(y)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


class TestFrenchNormalisation(unittest.TestCase):
    def test_suffix_stripped_from_core(self):
        n = normalize_name("Boulangerie Saint Honore SAS")
        self.assertNotIn("sas", n.core.split())
        self.assertIn("sas", n.full.split())  # full form retained for scoring

    def test_sarl_eurl_sci(self):
        for suffix in ["SARL", "EURL", "SCI", "SA", "SELARL", "SPA", "GIE"]:
            n = normalize_name(f"Atelier Lumiere {suffix}")
            self.assertNotIn(suffix.lower(), n.core.split(), suffix)

    def test_french_address_terms_survive(self):
        a = normalize_address("24 Rue Saint Honore, 75001 Paris CEDEX")
        self.assertIn("75001", a.postcodes)
        for tok in ["rue", "paris", "cedex"]:
            self.assertIn(tok, a.tokens, tok)

    def test_country_open_set_untouched(self):
        # 'st' stays street in addresses (US-majority call); French saint
        # handling is a documented limitation, not a silent remap.
        a = normalize_address("123 Main St, Springfield")
        self.assertIn("street", a.tokens)

    def test_french_thoroughfare_abbrevs(self):
        a = normalize_address("18 Bd de la Republique, 69002 Lyon")
        self.assertIn("boulevard", a.tokens)
        a = normalize_address("6 Imp des Lilas, 75019 Paris")
        self.assertIn("impasse", a.tokens)
        a = normalize_address("Route de Marseille, 13200 Arles")
        self.assertIn("route", a.tokens)


class TestPriorCorrection(unittest.TestCase):
    def test_identity(self):
        p = np.array([0.05, 0.5, 0.95])
        np.testing.assert_allclose(prior_correct_proba(p, 0.2, 0.2), p)

    def test_oversample_pushes_down(self):
        p = np.array([0.9, 0.5, 0.1])
        out = prior_correct_proba(p, 0.05, 0.2)
        self.assertTrue(bool((out < p).all()))

    def test_undersample_pushes_up(self):
        p = np.array([0.9, 0.5, 0.1])
        out = prior_correct_proba(p, 0.4, 0.1)
        self.assertTrue(bool((out > p).all()))

    def test_closed_form(self):
        p = np.array([0.9, 0.1, 0.5])
        F = (0.95 / 0.05) * (0.2 / 0.8)
        np.testing.assert_allclose(
            prior_correct_proba(p, 0.05, 0.2), p / (p + (1 - p) * F), atol=1e-12)

    def test_invalid_rates_passthrough(self):
        p = np.array([0.7])
        np.testing.assert_allclose(prior_correct_proba(p, 0.0, 0.2), p)


class TestTreeModels(unittest.TestCase):
    def _roundtrip(self, kind):
        cfg = PipelineConfig()
        cfg.model_type = kind
        X, y = make_xy()
        # duplicate rows to satisfy the 50-sample minimum
        X = np.tile(X, (12, 1))
        y = np.tile(y, 12)
        m = MatchingModel(cfg)
        rep = m.fit_batch(X, y, tau=0.1)
        self.assertTrue(m.kind.startswith("tree_"))
        proba = m.predict_proba(X)
        self.assertEqual(proba.shape, (len(y),))
        self.assertTrue(bool(((proba >= 0) & (proba <= 1)).all()))
        with tempfile.TemporaryDirectory() as tmp:
            mp = Path(tmp) / "model.json"
            m.save(mp)
            self.assertTrue((Path(tmp) / "model.booster.pkl").exists())
            m2 = MatchingModel.load(mp, cfg)
            # round-trip: kind, threshold, booster, predictions identical
            self.assertEqual(m2.kind, m.kind)
            self.assertEqual(m2.threshold, m.threshold)
            self.assertIsNotNone(m2._booster)
            np.testing.assert_allclose(m2.predict_proba(X), proba, atol=1e-9)
        return rep

    def test_hgb(self):
        self._roundtrip("hgb")

    def test_lightgbm(self):
        self._roundtrip("lightgbm")

    def test_xgboost(self):
        self._roundtrip("xgboost")

    def test_catboost(self):
        self._roundtrip("catboost")

    def test_insufficient_data_falls_back(self):
        cfg = PipelineConfig()
        cfg.model_type = "hgb"
        m = MatchingModel(cfg)
        rep = m.fit_batch(np.zeros((10, len(FEATURE_NAMES))), np.zeros(10, dtype=np.int64))
        self.assertEqual(m.kind, "heuristic")


class TestConfig(unittest.TestCase):
    def test_baseline_defaults_preserved(self):
        cfg = PipelineConfig()
        self.assertEqual(cfg.model_type, "sgd")
        self.assertFalse(cfg.prior_correct)
        self.assertEqual(cfg.df_cap, 60)
        self.assertEqual(cfg.max_candidates_per_row, 80)

    def test_threshold_grid_extended(self):
        grid = PipelineConfig().threshold_grid
        self.assertAlmostEqual(grid[0], 0.20)
        self.assertGreaterEqual(grid[-1], 0.90)
        self.assertTrue(all(b > a for a, b in zip(grid, grid[1:])))

    def test_macro_f05_singletons(self):
        self.assertEqual(macro_f05({"a": set()}, {"a": set()}), 1.0)
        self.assertEqual(macro_f05({"a": set()}, {"a": {"X"}}), 0.0)

    def test_macro_pr(self):
        from business_entity_resolution.src.model import macro_pr
        p, r = macro_pr({"a": {"X", "Y"}, "b": set()},
                        {"a": {"X", "Z"}, "b": set()})
        self.assertAlmostEqual(p, (0.5 + 1.0) / 2)
        self.assertAlmostEqual(r, (0.5 + 1.0) / 2)


class TestThreeWaySplit(unittest.TestCase):
    def _tiny_dataset(self, tmp):
        n = 60
        s1 = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
        s2 = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
        s3 = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
        gt = ["source1_entity_id\tmatched_entity_ids"]
        for i in range(n):
            s1.append(f"S1-{i}\tAcme Corporation {i}\t{i} Main Street 62701\tUS")
            s2.append(f"S2-{i}\tAcme Corp {i}\t{i} Main St 62701\tUS")
            s3.append(f"S3-{i}\tUnrelated Business {i}\t{i} Far Road 00000\tUS")
            gt.append(f"S1-{i}\tS2-{i}" if i % 3 else f"S1-{i}\t")
        d = Path(tmp) / "dataset" / "train"
        d.mkdir(parents=True)
        (d / "train_source1.tsv").write_text("\n".join(s1) + "\n", encoding="utf-8")
        (d / "train_source2.tsv").write_text("\n".join(s2) + "\n", encoding="utf-8")
        (d / "train_source3.tsv").write_text("\n".join(s3) + "\n", encoding="utf-8")
        (d / "train_ground_truth.tsv").write_text("\n".join(gt) + "\n", encoding="utf-8")
        return Path(tmp)

    def test_fit_val_test_disjoint_and_reported(self):
        from business_entity_resolution.src.config import PipelineConfig
        from business_entity_resolution.src.model import MatchingModel
        from business_entity_resolution.src.pipeline import train_on_train_split
        with tempfile.TemporaryDirectory() as tmp:
            data = self._tiny_dataset(tmp)
            cfg = PipelineConfig()
            cfg.train_rows = 30
            cfg.calib_rows = 15
            cfg.test_rows = 15
            cfg.workers = 1
            report, model = train_on_train_split(
                data / "dataset" / "train", cfg, MatchingModel(cfg))
            self.assertEqual(report["n_train_rows"], 30)
            self.assertEqual(report["n_val_rows"], 15)
            self.assertEqual(report["n_test_rows"], 15)
            self.assertIn("val_macro_f05", report)
            self.assertIn("test_macro_f05", report)
            self.assertIn("test_precision", report)
            self.assertIn("test_recall", report)
            # threshold locked from validation only
            self.assertEqual(model.threshold,
                             float(report["calibration"]["threshold"]))
            self.assertIn("blocking_recall_fit", report)


class TestPostprocess(unittest.TestCase):
    def test_gate_high_accepts_all(self):
        from business_entity_resolution.src.postprocess import evidence_gate
        X = np.zeros((2, len(FEATURE_NAMES)))
        out = evidence_gate(["A", "B"], [0.95, 0.80], X,
                            threshold=0.52, gate_high=0.90)
        self.assertEqual(out, ["A", "B"])

    def test_gate_drops_uncorroborated(self):
        from business_entity_resolution.src.postprocess import evidence_gate
        X = np.zeros((1, len(FEATURE_NAMES)))
        out = evidence_gate(["A"], [0.70], X,
                            threshold=0.52, gate_high=0.90)
        self.assertEqual(out, [])

    def test_gate_keeps_postcode_corroborated(self):
        from business_entity_resolution.src.postprocess import evidence_gate
        X = np.zeros((1, len(FEATURE_NAMES)))
        X[0, FEATURE_NAMES.index("postcode_match")] = 1.0
        out = evidence_gate(["A"], [0.70], X,
                            threshold=0.52, gate_high=0.90)
        self.assertEqual(out, ["A"])

    def test_gate_empty_in_empty_out(self):
        from business_entity_resolution.src.postprocess import evidence_gate
        X = np.zeros((0, len(FEATURE_NAMES)))
        self.assertEqual(evidence_gate([], [], X, threshold=0.52), [])


class TestPhoneticKeys(unittest.TestCase):
    def test_soundex_vectors(self):
        from business_entity_resolution.src.blocking import soundex
        self.assertEqual(soundex("Smith"), "S530")
        self.assertEqual(soundex("Smyth"), "S530")
        self.assertEqual(soundex("Ashcraft"), "A261")

    def test_keys_opt_in(self):
        from business_entity_resolution.src.blocking import keys_for
        plain = keys_for("acme corporation", "123 main street")
        phon = keys_for("acme corporation", "123 main street", phonetic=True)
        self.assertTrue(any(k.startswith(b"y:") for k in phon))
        self.assertFalse(any(k.startswith(b"y:") for k in plain))
        # phonetic keys are additive: every plain key survives
        self.assertTrue(set(plain) <= set(phon))

    def test_per_token_keys(self):
        from business_entity_resolution.src.blocking import keys_for
        plain = keys_for("acme corporation", "123 main street")
        pertok = keys_for("acme corporation", "123 main street", per_token=True)
        # per-token keys present only when opted in
        self.assertTrue(any(k.startswith(b"t:") for k in pertok))
        self.assertFalse(any(k.startswith(b"t:") for k in plain))
        # additive: plain keys survive
        self.assertTrue(set(plain) <= set(pertok))
        # "acme" is a content token >= 4 chars -> t: key exists
        # ("corporation" is stripped as a legal suffix, so only t:acme)
        self.assertIn(b"t:acme", pertok)
        # address-first key x: present
        self.assertTrue(any(k.startswith(b"x:") for k in pertok))

    def test_per_token_shared_rare_token(self):
        # two records sharing only a rare token (not longest/first/last)
        # become connected via the t: key
        from business_entity_resolution.src.blocking import keys_for
        a = keys_for("everest trading company", "1 main st", per_token=True)
        b = keys_for("ariaveoio everest", "2 main st", per_token=True)
        shared = set(a) & set(b)
        self.assertTrue(any(k.startswith(b"t:everest") for k in shared))

    def test_soft_cap_keeps_whole_groups(self):
        # a shared key group larger than budget: soft cap keeps it whole,
        # hard cap truncates to exactly the budget
        from business_entity_resolution.src.blocking import KeyIndex
        from business_entity_resolution.src.config import PipelineConfig
        from business_entity_resolution.src.store import store_from_rows
        cfg = PipelineConfig()
        cfg.df_cap = 100
        rows = [("S2-%d" % i, "acme corporation", "123 main street 62701", "US")
                for i in range(10)]
        st = store_from_rows(rows)
        ix = KeyIndex.build(st, cfg)
        n, a = "acme corporation", "123 main street 62701"
        hard = ix.candidates_for(n, a, max_candidates=4)
        self.assertEqual(len(hard), 4)
        cfg.soft_cap = True
        soft = ix.candidates_for(n, a, max_candidates=4)
        self.assertGreater(len(soft), 4)
        self.assertTrue(set(hard.tolist()) <= set(soft.tolist()))
        # determinism: same query twice -> identical
        again = ix.candidates_for(n, a, max_candidates=4)
        self.assertEqual(soft.tolist(), again.tolist())

    def test_minhash_bands(self):
        from business_entity_resolution.src.blocking import keys_for, minhash_bands
        # identical strings share all 16 bands (deterministic)
        a = minhash_bands("acme corporation")
        b = minhash_bands("acme corporation")
        self.assertEqual(len(a), 16)
        self.assertEqual(a, b)
        # one-char perturbation keeps a clear majority of bands vs unrelated
        c = minhash_bands("acme corporatiom")
        self.assertGreaterEqual(len(set(a) & set(c)), 6)
        # unrelated names share (near-)no bands
        d = minhash_bands("zxqv wmjk blorp")
        self.assertLessEqual(len(set(a) & set(d)), 2)
        # opt-in additive through keys_for
        plain = keys_for("acme corporation", "123 main street")
        mh = keys_for("acme corporation", "123 main street", minhash=True)
        self.assertTrue(any(k.startswith(b"h:") for k in mh))
        self.assertFalse(any(k.startswith(b"h:") for k in plain))
        self.assertTrue(set(plain) <= set(mh))
        self.assertEqual(minhash_bands("abc"), [])

    def _tiny_side(self):
        from business_entity_resolution.src.blocking import KeyIndex
        from business_entity_resolution.src.pipeline import TargetSide
        from business_entity_resolution.src.store import store_from_rows
        s1 = store_from_rows([
            ("S1-1", "acme corporation", "123 main street 62701", "US"),
            ("S1-2", "totally unique unfamiliar business name xyz",
             "7 remote village road 00000", "India"),
        ])
        s2 = store_from_rows([
            ("S2-1", "acme corp", "123 main st 62701", "US"),
            ("S2-2", "unrelated distant company", "999 far away 11111", "US"),
        ])
        cfg = PipelineConfig()
        side = TargetSide()
        side.stores.append(s2)
        side.indexes.append(KeyIndex.build(s2, cfg))
        return s1, side, cfg

    def test_score_chunk_baseline_matches(self):
        from business_entity_resolution.src.model import MatchingModel
        from business_entity_resolution.src.pipeline import score_chunk
        s1, side, cfg = self._tiny_side()
        model = MatchingModel(cfg)
        model._init_heuristic()
        preds, cands = score_chunk(model, s1, side, 0, s1.n, None, cfg)
        self.assertIn("S2-1", preds.get(0, []))
        self.assertNotIn("S2-2", preds.get(0, []))

    def test_score_chunk_flags_valid_and_deterministic(self):
        from business_entity_resolution.src.model import MatchingModel
        from business_entity_resolution.src.pipeline import score_chunk
        for flags in ({"evidence_gate": True},
                      {"graph_expand": True},
                      {"evidence_gate": True, "graph_expand": True}):
            s1, side, cfg = self._tiny_side()
            for k, v in flags.items():
                setattr(cfg, k, v)
            model = MatchingModel(cfg)
            model._init_heuristic()
            p1, c1 = score_chunk(model, s1, side, 0, s1.n, None, cfg)
            s1b, sideb, _ = self._tiny_side()
            p2, c2 = score_chunk(model, s1b, sideb, 0, s1b.n, None, cfg)
            self.assertEqual(p1, p2)
            self.assertEqual(c1, c2)
            for qi, keeps in p1.items():
                self.assertTrue(set(keeps) <= set(c1.get(qi, [])),
                                "matches must be a subset of candidates")


class TestSaveLoadRoundTrip(unittest.TestCase):
    """The 0.220 incident: load() must preserve kind/threshold/predictions."""

    def _rt(self, m, X, expect_booster):
        with tempfile.TemporaryDirectory() as tmp:
            mp = Path(tmp) / "model.json"
            m.save(mp)
            m2 = MatchingModel.load(mp, PipelineConfig())
            self.assertEqual(m2.kind, m.kind)
            self.assertEqual(m2.threshold, m.threshold)
            self.assertEqual(m2._booster is not None, expect_booster)
            np.testing.assert_allclose(m2.predict_proba(X),
                                       m.predict_proba(X), atol=1e-9)
        return m2

    def test_heuristic_roundtrip(self):
        cfg = PipelineConfig()
        m = MatchingModel(cfg)
        m._init_heuristic()
        X, _ = make_xy()
        m2 = self._rt(m, X, False)
        self.assertEqual(m2.kind, "heuristic")

    def test_logistic_roundtrip(self):
        cfg = PipelineConfig()
        m = MatchingModel(cfg)
        X, y = make_xy()
        X = np.tile(X, (12, 1))
        y = np.tile(y, 12)
        m.fit(X, y)
        self.assertEqual(m.kind, "logistic_regression")
        self._rt(m, X, False)

    def test_tree_kinds_preserved(self):
        for kind in ("hgb", "lightgbm", "xgboost", "catboost"):
            cfg = PipelineConfig()
            cfg.model_type = kind
            X, y = make_xy()
            X = np.tile(X, (12, 1))
            y = np.tile(y, 12)
            m = MatchingModel(cfg)
            m.fit_batch(X, y, tau=0.1)
            m.threshold = 0.96  # validated operating point must survive
            m2 = self._rt(m, X, True)
            self.assertTrue(m2.kind.startswith("tree_"))
            self.assertEqual(m2.threshold, 0.96)


if __name__ == "__main__":
    unittest.main()
