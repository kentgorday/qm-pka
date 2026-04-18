from pathlib import Path

import numpy as np
import pytest

from qm_pka.types import Conformer, Geometry
from qm_pka.xyz_io import read_multi_xyz, read_xyz, write_multi_xyz, write_xyz

WATER_XYZ = """\
3
water molecule
 O   0.0000000000   0.0000000000   0.1173000000
 H   0.0000000000   0.7572000000  -0.4692000000
 H   0.0000000000  -0.7572000000  -0.4692000000
"""

MULTI_XYZ = """\
3
     -76.4300000000
 O   0.0000000000   0.0000000000   0.1173000000
 H   0.0000000000   0.7572000000  -0.4692000000
 H   0.0000000000  -0.7572000000  -0.4692000000
3
     -76.4295000000
 O   0.0100000000   0.0000000000   0.1173000000
 H   0.0000000000   0.7600000000  -0.4692000000
 H   0.0000000000  -0.7600000000  -0.4692000000
"""


class TestReadXyz:
    def test_read_water(self, tmp_path: Path) -> None:
        p = tmp_path / "water.xyz"
        p.write_text(WATER_XYZ)
        geom = read_xyz(p)
        assert geom.symbols == ("O", "H", "H")
        assert geom.n_atoms == 3
        assert geom.coords.shape == (3, 3)
        np.testing.assert_allclose(geom.coords[0, 2], 0.1173)


class TestReadMultiXyz:
    def test_read_two_frames(self, tmp_path: Path) -> None:
        p = tmp_path / "ensemble.xyz"
        p.write_text(MULTI_XYZ)
        conformers = read_multi_xyz(p)
        assert len(conformers) == 2
        assert conformers[0].electronic_energy == pytest.approx(-76.43)
        assert conformers[1].electronic_energy == pytest.approx(-76.4295)
        assert conformers[0].geometry.symbols == ("O", "H", "H")


class TestWriteXyz:
    def test_round_trip(self, tmp_path: Path) -> None:
        geom = Geometry(
            symbols=("C", "H", "H", "H", "H"),
            coords=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.089, 0.0, 0.0],
                    [-0.363, 1.027, 0.0],
                    [-0.363, -0.513, 0.890],
                    [-0.363, -0.513, -0.890],
                ]
            ),
        )
        p = tmp_path / "methane.xyz"
        write_xyz(geom, p, comment="methane")
        geom2 = read_xyz(p)
        assert geom2.symbols == geom.symbols
        np.testing.assert_allclose(geom2.coords, geom.coords, atol=1e-8)


class TestWriteMultiXyz:
    def test_round_trip(self, tmp_path: Path) -> None:
        geom1 = Geometry(symbols=("H", "H"), coords=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]))
        geom2 = Geometry(symbols=("H", "H"), coords=np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]]))
        confs = [
            Conformer(geometry=geom1, electronic_energy=-1.17),
            Conformer(geometry=geom2, electronic_energy=-1.16),
        ]
        p = tmp_path / "h2.xyz"
        write_multi_xyz(confs, p)
        confs2 = read_multi_xyz(p)
        assert len(confs2) == 2
        assert confs2[0].electronic_energy == pytest.approx(-1.17)
        assert confs2[1].electronic_energy == pytest.approx(-1.16)
