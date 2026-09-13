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
