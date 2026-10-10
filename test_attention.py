import math
import unittest

import torch

from attention import flash_attn_func


class AttentionTests(unittest.TestCase):
    def test_full_and_windowed_grouped_attention_match_reference(self):
        torch.manual_seed(7)
        for heads, window in [(2, (-1, -1)), (1, (-1, -1)), (1, (2, 0))]:
            with self.subTest(heads=heads, window=window):
                q = torch.randn(2, 8, 2, 4, requires_grad=True)
                k = torch.randn(2, 8, heads, 4, requires_grad=True)
                v = torch.randn(2, 8, heads, 4, requires_grad=True)
                actual = flash_attn_func(q, k, v, window_size=window)
                repeat = 2 // heads
                kr = k.repeat_interleave(repeat, dim=2).transpose(1, 2)
                vr = v.repeat_interleave(repeat, dim=2).transpose(1, 2)
                scores = q.transpose(1, 2) @ kr.transpose(-1, -2) / math.sqrt(4)
                index = torch.arange(8)
                allowed = index[None, :] <= index[:, None]
                if window[0] >= 0:
                    allowed &= index[None, :] >= index[:, None] - window[0]
                expected = (scores.masked_fill(~allowed, float('-inf')).softmax(-1) @ vr).transpose(1, 2)
                torch.testing.assert_close(actual, expected)
                actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
                expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
