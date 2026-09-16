import json, sys
from pathlib import Path
src=Path(sys.argv[1]); dst=Path(sys.argv[2]); d=json.loads(src.read_text())
lines=["#pragma once","#include <array>","#include <cstddef>","namespace WarpTrackGeometry {",
"struct Vertex { double u; double z; };",
"struct Piece { int layer, bar, channelOffset, centerIndex; bool sensitive; std::array<Vertex,3> v; };",
f"inline constexpr double rackUnitMM={d['rack']['rack_unit_height']:.17g};",
f"inline constexpr double widthMM={d['rack']['width']:.17g};",
f"inline constexpr double depthMM={d['rack']['depth']:.17g};",
f"inline constexpr double hodoscopeHeightMM={d['hodoscope']['height']:.17g};",
f"inline constexpr int channelsPerHodoscope={d['hodoscope']['channels_per_hodoscope']};"]
inst=d['instances']; lines.append(f"inline constexpr std::array<double,{len(inst)}> rackU = {{{', '.join(str(x['rack_u']) for x in inst)}}};")
pieces=[p|{'layer':L['id'],'offset':L['channel_offset']} for L in d['layers'] for p in L['pieces']]
lines.append(f"inline constexpr std::array<Piece,{len(pieces)}> pieces = {{{{")
for p in pieces:
    vs=', '.join('Vertex{%s,%s}'%(v[0],v[1]) for v in p['vertices'])
    lines.append(f" Piece{{{p['layer']},{p['bar_id']},{p['offset']},{p['center_index']},{str(p['sensitive']).lower()}, {{{vs}}}}},")
lines += ["}};","}"]
dst.parent.mkdir(parents=True,exist_ok=True); dst.write_text('\n'.join(lines)+'\n')
