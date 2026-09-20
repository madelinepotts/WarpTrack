import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
d = json.loads(src.read_text())
server_model = d.get("server_model", {})
server_types = d.get("server_types", {})
servers = d.get("servers", [])
type_names = list(server_types)
type_index = {name: i for i, name in enumerate(type_names)}

lines = [
    "#pragma once", "#include <array>", "#include <cstddef>",
    "namespace WarpTrackGeometry {", "",
    "struct Vertex { double u; double z; };", "",
    "struct Piece { int layer; int bar; int channelOffset; int centerIndex; bool sensitive; std::array<Vertex, 3> v; };", "",
    "struct HodoscopeInstance { int id; double rackU; };", "",
    "struct ServerType { const char* name; int heightU; double widthMM; double depthMM; const char* chassisMaterial; double wallThicknessMM; double interiorDensityGCM3; };", "",
    "struct ServerInstance { int id; double rackU; int typeIndex; };", "",
    f"inline constexpr double rackUnitMM = {d['rack']['rack_unit_height']:.17g};",
    f"inline constexpr double widthMM = {d['rack']['width']:.17g};",
    f"inline constexpr double depthMM = {d['rack']['depth']:.17g};",
    f"inline constexpr double hodoscopeHeightMM = {d['hodoscope']['height']:.17g};",
    f"inline constexpr int channelsPerHodoscope = {d['hodoscope']['channels_per_hodoscope']};", "",
    f"inline constexpr bool serverModelEnabled = {str(bool(server_model.get('enabled', False))).lower()};", ""
]
instances=d['instances']
lines.append(f"inline constexpr std::array<HodoscopeInstance, {len(instances)}> hodoscopes = {{")
for x in instances: lines.append(f"  HodoscopeInstance{{{x['id']}, {float(x['rack_u']):.17g}}},")
lines += ["};", ""]
lines.append(f"inline constexpr std::array<ServerType, {len(type_names)}> serverTypes = {{")
for name in type_names:
    t=server_types[name]
    lines.append(f'  ServerType{{"{name}", {int(t["height_u"])}, {float(t["width"]):.17g}, {float(t["depth"]):.17g}, "{t["chassis_material"]}", {float(t["wall_thickness"]):.17g}, {float(t["interior_density_g_cm3"]):.17g}}},')
lines += ["};", ""]
lines.append(f"inline constexpr std::array<ServerInstance, {len(servers)}> servers = {{")
for s in servers: lines.append(f"  ServerInstance{{{s['id']}, {float(s['rack_u']):.17g}, {type_index[s['type']]}}},")
lines += ["};", ""]
pieces=[p|{'layer':L['id'],'offset':L['channel_offset']} for L in d['layers'] for p in L['pieces']]
lines.append(f"inline constexpr std::array<Piece, {len(pieces)}> pieces = {{{{")
for p in pieces:
    vs=', '.join(f"Vertex{{{float(v[0]):.17g}, {float(v[1]):.17g}}}" for v in p['vertices'])
    lines.append(f"  Piece{{{p['layer']}, {p['bar_id']}, {p['offset']}, {p['center_index']}, {str(p['sensitive']).lower()}, {{{vs}}}}},")
lines += ["}};", "", "}  // namespace WarpTrackGeometry"]
dst.parent.mkdir(parents=True,exist_ok=True)
dst.write_text('\n'.join(lines)+'\n')
