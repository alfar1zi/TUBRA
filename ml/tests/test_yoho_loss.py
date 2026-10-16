import pytest
import torch

from ml.src.yoho.model import PRESENCE_POSITIVE, yoho_loss


def _targets(batch, n_steps):
    presence = torch.zeros(batch, n_steps, 1)
    starts = torch.zeros(batch, n_steps, 1)
    ends = torch.zeros(batch, n_steps, 1)
    for row in range(batch):
        presence[row, 1, 0] = 1.0
        starts[row, 1, 0] = 0.4
        ends[row, 1, 0] = 0.6
    return presence, starts, ends


def _prediction(batch, n_steps):
    return torch.zeros(batch, 3, n_steps)


def test_perfect_prediction_is_zero():
    prediction = _prediction(1, 6)
    prediction[0, 0, 1] = 1.0
    prediction[0, 1, 1] = 0.4
    prediction[0, 2, 1] = 0.6
    presence, starts, ends = _targets(1, 6)
    assert float(yoho_loss(prediction, presence, starts, ends)) == pytest.approx(0.0)


def test_perfect_prediction_scales_with_total_steps():
    for n_steps in (6, 40):
        prediction = _prediction(1, n_steps)
        prediction[0, 0, 1] = 1.0
        prediction[0, 1, 1] = 0.4
        prediction[0, 2, 1] = 0.6
        presence, starts, ends = _targets(1, n_steps)
        value = float(yoho_loss(prediction, presence, starts, ends))
        assert value == pytest.approx(0.0)


def test_presence_error_on_positive_step_is_weighted():
    prediction = _prediction(1, 6)
    prediction[0, 0, 1] = 0.5
    prediction[0, 1, 1] = 0.4
    prediction[0, 2, 1] = 0.6
    presence, starts, ends = _targets(1, 6)
    weight = 5.0
    expected = 0.25 * weight / (5.0 + weight)
    value = float(yoho_loss(prediction, presence, starts, ends))
    assert value == pytest.approx(expected)


def test_absence_on_positive_step_is_weighted():
    prediction = _prediction(1, 6)
    prediction[0, 1, 1] = 0.4
    prediction[0, 2, 1] = 0.6
    presence, starts, ends = _targets(1, 6)
    weight = 5.0
    expected = weight / (5.0 + weight)
    value = float(yoho_loss(prediction, presence, starts, ends))
    assert value == pytest.approx(expected)


def test_presence_on_negative_step_is_counted_once():
    prediction = _prediction(1, 6)
    prediction[0, 0, 1] = 1.0
    prediction[0, 0, 3] = 0.5
    prediction[0, 1, 1] = 0.4
    prediction[0, 2, 1] = 0.6
    presence, starts, ends = _targets(1, 6)
    value = float(yoho_loss(prediction, presence, starts, ends))
    assert value == pytest.approx(0.25 / 10.0)


def test_start_offset_is_penalised():
    prediction = _prediction(1, 6)
    prediction[0, 0, 1] = 1.0
    prediction[0, 1, 1] = 0.0
    prediction[0, 2, 1] = 0.6
    presence, starts, ends = _targets(1, 6)
    value = float(yoho_loss(prediction, presence, starts, ends))
    assert value == pytest.approx(0.4 * 0.4 / 2.0)


def test_end_offset_is_penalised():
    prediction = _prediction(1, 6)
    prediction[0, 0, 1] = 1.0
    prediction[0, 1, 1] = 0.4
    prediction[0, 2, 1] = 0.0
    presence, starts, ends = _targets(1, 6)
    value = float(yoho_loss(prediction, presence, starts, ends))
    assert value == pytest.approx(0.6 * 0.6 / 2.0)


def test_offset_is_not_scored_on_negative_steps():
    prediction = _prediction(1, 6)
    prediction[0, 0, 1] = 1.0
    prediction[0, 1, 1] = 0.4
    prediction[0, 2, 1] = 0.6
    prediction[0, 1, 3] = 0.9
    prediction[0, 2, 3] = 0.9
    presence, starts, ends = _targets(1, 6)
    value = float(yoho_loss(prediction, presence, starts, ends))
    assert value == pytest.approx(0.0)


def test_swapping_presence_and_start_planes_is_penalised():
    prediction = _prediction(1, 6)
    prediction[0, 0, 1] = 0.4
    prediction[0, 1, 1] = 1.0
    prediction[0, 2, 1] = 0.6
    presence, starts, ends = _targets(1, 6)
    value = float(yoho_loss(prediction, presence, starts, ends))
    weight = 5.0
    expected = 0.36 * weight / (5.0 + weight)
    expected = expected + (0.6 * 0.6 + 0.0) / 2.0
    assert value == pytest.approx(expected)
    assert value > 0.1


def test_positive_weight_is_capped():
    n_steps = 40
    prediction = _prediction(1, n_steps)
    presence, starts, ends = _targets(1, n_steps)
    presence[0, 39, 0] = 1.0
    starts[0, 39, 0] = 0.4
    ends[0, 39, 0] = 0.6
    prediction[0, 0, 1] = 1.0
    prediction[0, 1, 1] = 0.4
    prediction[0, 2, 1] = 0.6
    expected = (38.0 / 2.0) / (38.0 + 38.0 / 2.0)
    value = float(yoho_loss(prediction, presence, starts, ends))
    assert value == pytest.approx(expected, rel=0.2)
    assert value > 0.0


def test_batch_averages_over_examples():
    prediction = _prediction(2, 6)
    prediction[0, 0, 1] = 1.0
    prediction[0, 1, 1] = 0.4
    prediction[0, 2, 1] = 0.6
    prediction[1, 0, 1] = 1.0
    prediction[1, 1, 1] = 0.4
    presence, starts, ends = _targets(2, 6)
    value = float(yoho_loss(prediction, presence, starts, ends))
    assert value == pytest.approx(0.6 * 0.6 / 2.0 / 2.0)


@pytest.mark.parametrize("batch", [1, 2, 32])
def test_batch_matches_single_example(batch):
    n_steps = 6
    prediction = _prediction(batch, n_steps)
    for row in range(batch):
        prediction[row, 0, 1] = 1.0
        prediction[row, 1, 1] = 0.4
        prediction[row, 2, 1] = 0.6
        prediction[row, 0, 1] = prediction[row, 0, 1] - row * 0.01
    presence, starts, ends = _targets(batch, n_steps)
    batched = float(yoho_loss(prediction, presence, starts, ends))
    single = []
    for row in range(batch):
        one = prediction[row : row + 1]
        single.append(
            float(
                yoho_loss(
                    one,
                    presence[row : row + 1],
                    starts[row : row + 1],
                    ends[row : row + 1],
                )
            )
        )
    assert batched == pytest.approx(sum(single) / batch)


def test_shape_mismatch_raises():
    prediction = _prediction(1, 6)
    presence, starts, ends = _targets(1, 6)
    with pytest.raises(ValueError):
        yoho_loss(torch.zeros(1, 6, 6), presence, starts, ends)
    with pytest.raises(ValueError):
        yoho_loss(torch.zeros(6, 3), presence, starts, ends)


def test_negative_threshold_constant():
    assert 0.0 < PRESENCE_POSITIVE < 0.5
