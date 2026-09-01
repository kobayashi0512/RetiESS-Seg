import unittest

import torch
from torch import nn

from retiess import ESSLogitShrinkage, RetiESSPool


class RetiESSPoolTests(unittest.TestCase):
    def test_gradients_flow_through_pooling_and_kernels(self):
        torch.manual_seed(7)
        features = torch.randn(3, 8, 6, requires_grad=True)
        pool = RetiESSPool(
            feature_dim=6,
            attention_hidden_dim=12,
            num_iterations=8,
        )
        classifier = nn.Linear(6, 2)

        output = pool(features)
        raw_logits = classifier(output.pooled)
        logits = ESSLogitShrinkage(prior_logits=torch.zeros(2))(
            raw_logits, output.n_eff
        )
        loss = logits.square().mean() + 0.01 * output.n_eff.mean()
        loss.backward()

        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0.0)
        scorer_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in pool.scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(scorer_grad, 0.0)

    def test_independent_slices_have_large_effective_sample_size(self):
        num_slices = 8
        features = torch.eye(num_slices).unsqueeze(0)
        positions = torch.arange(num_slices, dtype=torch.float32)
        scores = torch.zeros(1, num_slices)
        pool = RetiESSPool(
            feature_dim=num_slices,
            positional_mix=0.5,
            position_bandwidth=0.01,
            feature_bandwidth=0.05,
            num_iterations=8,
        )

        output = pool(features, positions=positions, scores=scores)

        self.assertAlmostEqual(output.weights.sum().item(), 1.0, places=6)
        self.assertGreater(output.n_eff.item(), num_slices - 1e-3)

    def test_highly_repeated_slices_have_effective_sample_size_near_one(self):
        num_slices = 8
        features = torch.ones(1, num_slices, 5)
        positions = torch.zeros(num_slices)
        scores = torch.zeros(1, num_slices)
        pool = RetiESSPool(
            feature_dim=5,
            positional_mix=0.5,
            num_iterations=8,
        )

        output = pool(features, positions=positions, scores=scores)

        self.assertTrue(
            torch.allclose(output.correlation, torch.ones_like(output.correlation))
        )
        self.assertAlmostEqual(output.n_eff.item(), 1.0, places=5)

    def test_low_effective_size_causes_stronger_prior_shrinkage(self):
        shrink = ESSLogitShrinkage(prior_logits=0.0, kappa=4.0)
        logits = torch.tensor([[4.0], [4.0]])
        n_eff = torch.tensor([1.0, 16.0])

        adjusted = shrink(logits, n_eff)

        self.assertLess(adjusted[0].abs().item(), adjusted[1].abs().item())
        self.assertAlmostEqual(adjusted[0].item(), 0.8, places=6)
        self.assertAlmostEqual(adjusted[1].item(), 3.2, places=6)


if __name__ == "__main__":
    unittest.main()
