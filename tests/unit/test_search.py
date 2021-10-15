# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base._tests import PyironTestCase
from pyiron_ising.search import bfs, double_bfs
import numpy as np


class TestBFS(PyironTestCase):

    @classmethod
    def setUpClass(cls):
        """
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15

        x x x o
        x x o o
        x x x o
        x x x o
        """
        super().setUpClass()
        l = 4
        cls.nodes = np.arange(l * l, dtype=int)
        nodegrid = np.reshape(cls.nodes, (l, l))
        cls.topology = np.stack(
            (
                np.roll(nodegrid, 1, axis=1),
                np.roll(nodegrid, -1, axis=1),
                np.roll(nodegrid, 1, axis=0),
                np.roll(nodegrid, -1, axis=0),
            ),
            axis=-1
        ).reshape(l * l, -1)

        cls.signature_ = np.array('x x x o x x o o x x x o x x x o'.split())

    @staticmethod
    def condition(i, j, topo, sig, thresh):
        return (sig[i] == sig[j]) and (np.sum(sig[topo[j]] == sig[j]) >= thresh)

    def test_bfs(self):
        self.assertCountEqual(
            [9, 13, 1],
            bfs(9, self.topology, self.condition, topo=self.topology, sig=self.signature_, thresh=4),
            msg="Should only get x's completely surrounded by x's. Don't forget we have periodic boundary conditions."
        )
        self.assertCountEqual(
            self.nodes[self.signature_ == 'x'].tolist(),
            bfs(9, self.topology, self.condition, topo=self.topology, sig=self.signature_, thresh=0),
            msg="With no threshold, should get all nodes with the same signature."
        )
        self.assertCountEqual(
            [3, 7, 11, 15],
            bfs(7, self.topology, self.condition, topo=self.topology, sig=self.signature_, thresh=2),
            msg="Should be getting that righthand column, minus the nub at 6."
        )
        self.assertEqual(
            1, len(bfs(0, self.topology, self.condition, topo=self.topology, sig=self.signature_, thresh=42)),
            msg="If initial site does not meet the condition, it should be returned alone."
        )

    def test_double_bfs(self):
        for thresh in np.arange(5):
            for i in np.argwhere(self.signature_ == 'x'):
                for j in np.argwhere(self.signature_ == 'y'):
                    ci, cj = double_bfs(
                        i, j, self.topology, self.condition, topo=self.topology, sig=self.signature_, thresh=thresh
                    )
                    self.assertEqual(len(ci), len(cj), msg='Clusters should always be truncated to have same length.')
