"""Reject corrupt localization before it can become patch evidence."""
import pytest
from opendm.model.dm05.qwen_patch_scorer import validate_grounding

def test_multiple_real_objects_of_same_class_are_allowed():
    validate_grounding([dict(label="cup",bbox=[0,0,100,100]),dict(label="cup",bbox=[300,0,400,100])])

def test_repeated_full_image_boxes_are_rejected():
    with pytest.raises(ValueError,match="duplicate"):
        validate_grounding([dict(label="robot arm",bbox=[0,0,1000,1000]),dict(label="table",bbox=[0,0,1000,1000])])

@pytest.mark.parametrize("box",[[0,0,1001,100],[1,0,0,100],[True,0,10,20],[0,1,5,1]])
def test_bad_coordinates_are_rejected(box):
    with pytest.raises(ValueError):
        validate_grounding([dict(label="object",bbox=box)])

def test_unbounded_output_is_rejected():
    with pytest.raises(ValueError):
        validate_grounding([dict(label="object",bbox=[i,0,i+1,1]) for i in range(33)])

def test_feature_reuse_preserves_all_observation_times():
    from opendm.model.dm05.qwen_patch_scorer import observation_times
    context=[dict(frame=7,time=0.),dict(frame=7,time=5.),dict(frame=9,time=2.),dict(frame=7,time=0.)]
    candidates=[dict(frame=7,time=5.),dict(frame=9,time=2.)]
    expected={7:[0.,5.],9:[2.]}
    assert observation_times([7,9],context,candidates)==expected
    assert observation_times([7,9],list(reversed(context)),candidates)==expected
    with pytest.raises(ValueError,match="omits candidate"):
        observation_times([7,9],[dict(frame=7,time=0.),dict(frame=9,time=2.)],candidates)
