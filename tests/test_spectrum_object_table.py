import numpy as np
from PIL import Image

from tools.analysisTools.extractSpectrumObjectTable import extract_table


def test_extracts_finite_scene_aware_object_features(tmp_path):
    image = np.zeros((64, 64), dtype=np.uint8)
    image[20:28, 20:28] = 220
    image_path = tmp_path / 'scene01_tile01.png'
    Image.fromarray(image).save(image_path)
    annotations = dict(
        images=[dict(id=1, file_name=image_path.name, scene_id='scene01')],
        annotations=[dict(
            id=11, image_id=1, category_id=3, bbox=[20, 20, 8, 8], iscrowd=0
        )],
    )
    predictions = [dict(
        image_id=1, category_id=3, bbox=[20, 20, 8, 8], score=0.9
    )]
    table = extract_table(annotations, predictions, tmp_path)
    assert len(table) == 1
    assert table.loc[0, 'scene_id'] == 'scene01'
    assert table.loc[0, 'target_error'] == 0.0
    assert np.isfinite(table[[
        'contrast', 'background_complexity',
        'll_energy', 'lh_energy', 'hl_energy', 'hh_energy',
    ]].to_numpy()).all()
    assert table.loc[0, 'contrast'] > 0


def test_unmatched_object_receives_maximum_predefined_error(tmp_path):
    image_path = tmp_path / 'tile.png'
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(image_path)
    annotations = dict(
        images=[dict(id=1, file_name=image_path.name, scene_id='scene')],
        annotations=[dict(
            id=1, image_id=1, category_id=1, bbox=[8, 8, 4, 4], iscrowd=0
        )],
    )
    table = extract_table(annotations, [], tmp_path)
    assert table.loc[0, 'target_error'] == 1.0
