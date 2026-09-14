import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "metadata" / "cfd_evidence_store.json"

class CFDEvidenceStore:
    """
    Persistent store for all OpenFOAM CFD ground-truth runs.
    Provides methods to query validated points, compute empirical correction factors,
    and extract falsified gradient vectors for active constraint regularization.
    """
    def __init__(self, store_path: Optional[str] = None):
        self.store_path = Path(store_path).resolve() if store_path else DEFAULT_STORE_PATH
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.store_path.exists():
            return {
                "version": "2.0",
                "description": "AeroMorphs Phase 7 & 8 CFD Ground-Truth Evidence Store",
                "entries": []
            }
        try:
            with open(self.store_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load evidence store ({e}). Initializing empty store.")
            return {
                "version": "2.0",
                "description": "AeroMorphs Phase 7 & 8 CFD Ground-Truth Evidence Store",
                "entries": []
            }

    def save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def add_entry(self, entry: Dict[str, Any]):
        entries = self.data.setdefault("entries", [])
        for i, existing in enumerate(entries):
            if existing.get("id") == entry.get("id"):
                entries[i] = entry
                self.save()
                return
        entries.append(entry)
        self.save()

    def get_entries(self, category: Optional[str] = None, body_type: Optional[str] = None) -> List[Dict[str, Any]]:
        results = self.data.get("entries", [])
        if category:
            results = [e for e in results if e.get("category") == category]
        if body_type:
            results = [e for e in results if e.get("body_type") == body_type]
        return results

    def get_directional_constraints(
        self, 
        body_type: Optional[str] = None,
        baseline_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of falsified gradient vectors where CFD proved the surrogate's
        predicted reduction was an illusion (i.e. Delta CdA_CFD > Delta CdA_surrogate).
        """
        constraints = []
        for entry in self.data.get("entries", []):
            if body_type and entry.get("body_type") != body_type:
                continue
            if baseline_id and entry.get("baseline_id") != baseline_id:
                continue
            z_init_path = entry.get("z_initial_path")
            z_opt_path = entry.get("z_opt_path")
            
            if not z_init_path or not z_opt_path:
                continue
            if not os.path.exists(z_init_path) or not os.path.exists(z_opt_path):
                continue
                
            delta_cfd = entry.get("delta_cda_cfd")
            delta_surr = entry.get("delta_cda_surrogate")
            
            if delta_cfd is not None and delta_surr is not None:
                discrepancy = delta_cfd - delta_surr
                # If CFD was worse than predicted, this is a falsified direction
                if discrepancy > 0:
                    z0 = torch.load(z_init_path, map_location="cpu").squeeze()
                    z1 = torch.load(z_opt_path, map_location="cpu").squeeze()
                    v = z1 - z0
                    norm_v = torch.norm(v).item()
                    if norm_v > 1e-6:
                        constraints.append({
                            "id": entry.get("id"),
                            "v_norm": v / norm_v,
                            "v": v,
                            "norm_v": norm_v,
                            "error_delta": discrepancy,
                            "z_initial": z0,
                            "z_opt": z1
                        })
        return constraints

    def get_baseline_entry(self, baseline_id: str) -> Optional[Dict[str, Any]]:
        """Find baseline entry for a given vehicle id."""
        for entry in self.data.get("entries", []):
            cat = entry.get("category", "")
            if "Baseline" in cat:
                if entry.get("baseline_id") == baseline_id or entry.get("id") == baseline_id:
                    return entry
        return None

    def record_run(
        self,
        run_id: str,
        category: str,
        body_type: str,
        baseline_id: str,
        cfd_results: Dict[str, Any],
        opt_summary: Optional[Dict[str, Any]] = None,
        z_initial_path: Optional[str] = None,
        z_opt_path: Optional[str] = None,
        round_num: Optional[int] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Automated ingestion of a CFD run result into the evidence store.
        Calculates discrepancies, deltas against baseline, and updates persistent JSON.
        """
        cfd_cda = cfd_results.get("cda_m2") if cfd_results.get("cda_m2") is not None else cfd_results.get("cfd_cda")
        cfd_force = cfd_results.get("mean_drag_force_N") if cfd_results.get("mean_drag_force_N") is not None else cfd_results.get("cfd_drag_force_N")
        cfd_std = cfd_results.get("std_drag_force_N") if cfd_results.get("std_drag_force_N") is not None else cfd_results.get("cfd_std_N")
        cells = cfd_results.get("cells") or cfd_results.get("mesh_cells")

        baseline_entry = self.get_baseline_entry(baseline_id)
        delta_cda_cfd = None
        delta_drag_force_N = None
        real_drag_change_pct = None

        if baseline_entry:
            base_cda = baseline_entry.get("cfd_cda")
            base_force = baseline_entry.get("cfd_drag_force_N")
            if cfd_cda is not None and base_cda is not None:
                delta_cda_cfd = cfd_cda - base_cda
            if cfd_force is not None and base_force is not None:
                delta_drag_force_N = cfd_force - base_force
                if base_force != 0:
                    real_drag_change_pct = (delta_drag_force_N / base_force) * 100.0

        surrogate_pred_cda = None
        delta_cda_surrogate = None
        trust_radius = None

        if opt_summary:
            surrogate_pred_cda = opt_summary.get("final_predicted_drag_area")
            initial_pred = opt_summary.get("baseline_predicted_drag_area")
            if surrogate_pred_cda is not None and initial_pred is not None:
                delta_cda_surrogate = surrogate_pred_cda - initial_pred
            trust_radius = opt_summary.get("trust_radius")
            if not z_initial_path:
                z_initial_path = opt_summary.get("z_initial_path")
            if not z_opt_path:
                z_opt_path = opt_summary.get("z_opt_path")

        entry: Dict[str, Any] = {
            "id": run_id,
            "category": category,
            "body_type": body_type,
            "baseline_id": baseline_id,
        }
        if round_num is not None:
            entry["round"] = round_num
        if surrogate_pred_cda is not None:
            entry["surrogate_pred_cda"] = round(float(surrogate_pred_cda), 5)
        if delta_cda_surrogate is not None:
            entry["delta_cda_surrogate"] = round(float(delta_cda_surrogate), 5)
        if cfd_force is not None:
            entry["cfd_drag_force_N"] = round(float(cfd_force), 4)
        if cfd_cda is not None:
            entry["cfd_cda"] = round(float(cfd_cda), 5)
        if delta_cda_cfd is not None:
            entry["delta_cda_cfd"] = round(float(delta_cda_cfd), 5)
        if delta_drag_force_N is not None:
            entry["delta_drag_force_N"] = round(float(delta_drag_force_N), 4)
        if real_drag_change_pct is not None:
            entry["real_drag_change_vs_baseline_pct"] = round(float(real_drag_change_pct), 2)
        if cfd_std is not None:
            entry["cfd_std_N"] = round(float(cfd_std), 4)
        if cells is not None:
            entry["cells"] = int(cells)
        if z_initial_path:
            entry["z_initial_path"] = str(Path(z_initial_path).resolve())
        if z_opt_path:
            entry["z_opt_path"] = str(Path(z_opt_path).resolve())
        if trust_radius is not None:
            entry["trust_radius"] = float(trust_radius)
        if notes:
            entry["notes"] = notes

        self.add_entry(entry)
        
        disc_str = ""
        if delta_cda_cfd is not None and delta_cda_surrogate is not None:
            discrepancy = delta_cda_cfd - delta_cda_surrogate
            disc_str = f" | Discrepancy: {discrepancy:+.4f} m^2 ({'FALSIFIED (barrier active)' if discrepancy > 0 else 'VERIFIED'})"
        print(f"[EvidenceStore] Recorded entry '{run_id}' (CFD CdA: {cfd_cda:.4f} m^2{disc_str})")
        return entry

