import json
import unittest
from pathlib import Path


class ServerGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "geometry" / "detector_geometry.json"
        cls.geometry = json.loads(path.read_text())

    def test_server_model_is_enabled(self):
        self.assertTrue(self.geometry["server_model"]["enabled"])

    def test_server_types_support_1u_and_2u(self):
        types = self.geometry["server_types"]
        self.assertEqual(types["generic_1u"]["height_u"], 1)
        self.assertEqual(types["generic_2u"]["height_u"], 2)
        for server_type in types.values():
            self.assertIn(server_type["chassis_material"], {"G4_Al", "G4_STAINLESS-STEEL"})
            self.assertGreater(server_type["wall_thickness"], 0.0)
            self.assertGreater(server_type["interior_density_g_cm3"], 0.0)

    def test_servers_reference_valid_types(self):
        types = self.geometry["server_types"]
        ids = set()
        for server in self.geometry["servers"]:
            self.assertIn(server["type"], types)
            self.assertNotIn(server["id"], ids)
            ids.add(server["id"])

    def test_servers_do_not_overlap_hodoscopes_or_each_other(self):
        u_mm = self.geometry["rack"]["rack_unit_height"]
        h_h = self.geometry["hodoscope"]["height"] / 2.0
        occupied = []
        for h in self.geometry["instances"]:
            z = h["rack_u"] * u_mm
            occupied.append((z - h_h, z + h_h, f"hodoscope {h['id']}"))
        for s in self.geometry["servers"]:
            t = self.geometry["server_types"][s["type"]]
            z = s["rack_u"] * u_mm
            half = t["height_u"] * u_mm / 2.0
            occupied.append((z - half, z + half, f"server {s['id']}"))
        occupied.sort()
        for a, b in zip(occupied, occupied[1:]):
            self.assertLessEqual(a[1], b[0], f"{a[2]} overlaps {b[2]}")


if __name__ == "__main__":
    unittest.main()
