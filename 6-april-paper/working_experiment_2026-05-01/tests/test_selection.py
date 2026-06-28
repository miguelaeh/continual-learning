import torch

from smf_retrofit.continual.selection import KLSlotSelector, TFIDFSlotSelector


def test_tfidf_prefers_rare_slots():
    df = torch.zeros(64, dtype=torch.long)
    df[:32] = 100
    df[40] = 1
    selector = TFIDFSlotSelector(df, total_batches=200, top_t=4)

    access = torch.zeros(64, dtype=torch.long)
    access[5] = 10
    access[40] = 10
    chosen = selector.select(access)

    assert 40 in chosen.tolist()


def test_kl_prefers_surprising_slots():
    df = torch.ones(64, dtype=torch.long) * 50
    df[50] = 1
    selector = KLSlotSelector(df, total_batches=100, top_t=4)

    access = torch.zeros(64, dtype=torch.long)
    access[10] = 10
    access[50] = 10
    chosen = selector.select(access)

    assert 50 in chosen.tolist()
