"""Regression checks for the TimeMixer-style vector composition."""
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

from paper.draw_framework_timemixer import draw

ROOT = Path(__file__).resolve().parent


class TimeMixerFigureTests(unittest.TestCase):
    def test_text_is_inside_canvas_and_does_not_overlap(self):
        with np.load(ROOT / "figure_data/framework_hierarchy.npz", allow_pickle=False) as data:
            fig = draw(data)
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in fig._diagram_labels]
            for label, bounds in boxes:
                self.assertGreaterEqual(bounds.x0, -1, label)
                self.assertGreaterEqual(bounds.y0, -1, label)
                self.assertLessEqual(bounds.x1, fig.bbox.width+1, label)
                self.assertLessEqual(bounds.y1, fig.bbox.height+1, label)
            for index, (first, a) in enumerate(boxes):
                for second, b in boxes[index+1:]:
                    self.assertFalse(a.overlaps(b), f"Overlapping labels: {first!r}, {second!r}")
        finally:
            plt.close(fig)

    def test_export_is_vector_and_font_is_embedded(self):
        svg = ET.parse(ROOT / "figures/framework_timemixer.svg")
        self.assertFalse(svg.findall(".//{http://www.w3.org/2000/svg}image"))
        self.assertGreater(len(svg.findall(".//{http://www.w3.org/2000/svg}text")), 40)
        pdf = (ROOT / "figures/framework_timemixer.pdf").read_bytes()
        self.assertNotIn(b"/Subtype /Image", pdf)
        self.assertIn(b"/FontFile2", pdf)
        self.assertIn(b"Inter", pdf)


if __name__ == "__main__":
    unittest.main()
