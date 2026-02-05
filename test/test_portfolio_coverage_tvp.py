import os
import unittest

import pandas as pd

from SBTi.portfolio_aggregation import PortfolioAggregationMethod
from SBTi.portfolio_coverage_tvp import PortfolioCoverageTVP
from SBTi.interfaces import FINZAlignmentCategory


class TestPortfolioCoverageTVP(unittest.TestCase):
    """
    Test the TVP portfolio coverage (checking which companies have a valid SBTi approved target.
    """

    def setUp(self) -> None:
        self.portfolio_coverage_tvp = PortfolioCoverageTVP()
        self.data = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "inputs",
                "portfolio_coverage.csv",
            ),
            encoding="iso-8859-1",
        )

    def test_coverage(self) -> None:
        """
        Test whether portfolio coverage is calculated correctly.
        """
        coverage = self.portfolio_coverage_tvp.get_portfolio_coverage(
            self.data, PortfolioAggregationMethod.WATS
        )
        self.assertAlmostEqual(
            coverage, 32.0663, places=4, msg="The portfolio coverage was not correct"
        )


class TestFINZAlignmentCategory(unittest.TestCase):
    """
    Test the FINZ alignment category enum values.
    """

    def test_enum_values(self) -> None:
        """
        Test that the enum has the expected values per FINZ Standard Table 4.2.
        For FINZ: only In Transition (1.5°C) and Assessed (everything else).
        """
        self.assertEqual(FINZAlignmentCategory.IN_TRANSITION.value, "In Transition")
        self.assertEqual(FINZAlignmentCategory.ASSESSED.value, "Assessed")

        # Verify there's no "Other SBT" category for FINZ
        self.assertEqual(len(FINZAlignmentCategory), 2)


class TestFINZPortfolioCoverage(unittest.TestCase):
    """
    Test the FINZ alignment assessment functionality (no weighting).
    """

    def setUp(self) -> None:
        """
        Create test data with target classification column.
        """
        self.portfolio_coverage_tvp = PortfolioCoverageTVP()

        # Test data with various target classifications
        self.data = pd.DataFrame({
            'company_name': ['Company A', 'Company B', 'Company C', 'Company D', 'Company E'],
            'company_id': ['A', 'B', 'C', 'D', 'E'],
            'investment_value': [100000, 200000, 150000, 50000, 500000],  # Total: 1,000,000
            'sbti_validated': [True, True, True, False, True],
            'target_classification': ['1.5°C', 'Well-below 2°C', '1.5°C/1.5°C', None, '2°C'],
        })

    def test_finz_categories_only_two(self) -> None:
        """
        Test that FINZ only has two categories: In Transition and Assessed.
        WB2°C and 2°C targets should be categorized as Assessed for FINZ.
        """
        result = self.portfolio_coverage_tvp.get_finz_portfolio_coverage(self.data)
        df = result['company_data']

        # Company A (1.5°C) should be In Transition
        self.assertEqual(
            df[df['company_id'] == 'A']['finz_category'].iloc[0],
            FINZAlignmentCategory.IN_TRANSITION.value
        )

        # Company B (WB2°C) should be Assessed for FINZ (not Other SBT)
        self.assertEqual(
            df[df['company_id'] == 'B']['finz_category'].iloc[0],
            FINZAlignmentCategory.ASSESSED.value
        )

        # Company C (1.5°C/1.5°C) should be In Transition
        self.assertEqual(
            df[df['company_id'] == 'C']['finz_category'].iloc[0],
            FINZAlignmentCategory.IN_TRANSITION.value
        )

        # Company D (no target) should be Assessed
        self.assertEqual(
            df[df['company_id'] == 'D']['finz_category'].iloc[0],
            FINZAlignmentCategory.ASSESSED.value
        )

        # Company E (2°C) should be Assessed for FINZ
        self.assertEqual(
            df[df['company_id'] == 'E']['finz_category'].iloc[0],
            FINZAlignmentCategory.ASSESSED.value
        )

    def test_coverage_calculation_no_weighting(self) -> None:
        """
        Test that coverage is calculated with simple sums (no weighting).
        """
        result = self.portfolio_coverage_tvp.get_finz_portfolio_coverage(self.data)
        coverage = result['coverage']

        # In Transition: Company A ($100k) + Company C ($150k) = $250k = 25%
        self.assertEqual(coverage['in_transition']['value'], 250000)
        self.assertAlmostEqual(coverage['in_transition']['value_pct'], 25.0, places=1)

        # In Transition count: 2 out of 5 = 40%
        self.assertEqual(coverage['in_transition']['count'], 2)
        self.assertAlmostEqual(coverage['in_transition']['count_pct'], 40.0, places=1)

        # Assessed: Company B + D + E = $200k + $50k + $500k = $750k = 75%
        self.assertEqual(coverage['assessed']['value'], 750000)
        self.assertAlmostEqual(coverage['assessed']['value_pct'], 75.0, places=1)

        # Assessed count: 3 out of 5 = 60%
        self.assertEqual(coverage['assessed']['count'], 3)
        self.assertAlmostEqual(coverage['assessed']['count_pct'], 60.0, places=1)

        # Totals
        self.assertEqual(coverage['total']['value'], 1000000)
        self.assertEqual(coverage['total']['count'], 5)

    def test_summary_statistics(self) -> None:
        """
        Test that summary provides high-level metrics.
        """
        result = self.portfolio_coverage_tvp.get_finz_portfolio_coverage(self.data)
        summary = result['summary']

        self.assertAlmostEqual(summary['finz_in_transition_value_pct'], 25.0, places=1)
        self.assertAlmostEqual(summary['finz_in_transition_count_pct'], 40.0, places=1)
        self.assertEqual(summary['total_portfolio_value'], 1000000)
        self.assertEqual(summary['total_companies'], 5)


class TestPortfolioCoverageWithAlignment(unittest.TestCase):
    """
    Test the backward-compatible alignment assessment functionality.
    """

    def setUp(self) -> None:
        """
        Create test data with target classification column.
        """
        self.portfolio_coverage_tvp = PortfolioCoverageTVP()

        # Test data with various target classifications
        self.data_with_classification = pd.DataFrame({
            'company_name': ['Company A', 'Company B', 'Company C', 'Company D'],
            'company_id': ['A', 'B', 'C', 'D'],
            'investment_value': [100000, 200000, 150000, 50000],
            'sbti_validated': [True, True, True, False],
            'target_classification': ['1.5°C', 'Well-below 2°C', '1.5°C/1.5°C', None],
        })

    def test_finz_alignment_categories(self) -> None:
        """
        Test that companies are correctly categorized into FINZ alignment categories.
        """
        result = self.portfolio_coverage_tvp.get_portfolio_coverage_with_alignment(
            self.data_with_classification,
            PortfolioAggregationMethod.WATS
        )

        df = result['company_data']

        # Company A (1.5°C) should be In Transition
        self.assertEqual(
            df[df['company_id'] == 'A']['finz_category'].iloc[0],
            FINZAlignmentCategory.IN_TRANSITION.value
        )

        # Company B (WB2°C) should be Assessed for FINZ
        self.assertEqual(
            df[df['company_id'] == 'B']['finz_category'].iloc[0],
            FINZAlignmentCategory.ASSESSED.value
        )

        # Company C (1.5°C/1.5°C) should be In Transition
        self.assertEqual(
            df[df['company_id'] == 'C']['finz_category'].iloc[0],
            FINZAlignmentCategory.IN_TRANSITION.value
        )

        # Company D (no target) should be Assessed
        self.assertEqual(
            df[df['company_id'] == 'D']['finz_category'].iloc[0],
            FINZAlignmentCategory.ASSESSED.value
        )

    def test_fint_vs_finz_coverage(self) -> None:
        """
        Test that FINT coverage includes all validated targets while FINZ only counts 1.5°C.
        """
        result = self.portfolio_coverage_tvp.get_portfolio_coverage_with_alignment(
            self.data_with_classification,
            PortfolioAggregationMethod.WATS
        )

        # FINT should include Company A, B, C (all validated)
        # FINZ should only include Company A, C (1.5°C targets)
        fint_pct = result['fint_coverage_pct']
        finz_pct = result['finz_in_transition_pct']

        # FINZ coverage should be less than or equal to FINT coverage
        self.assertLessEqual(finz_pct, fint_pct)

        # Both should be greater than 0 (we have validated companies)
        self.assertGreater(fint_pct, 0)
        self.assertGreater(finz_pct, 0)

    def test_summary_counts(self) -> None:
        """
        Test that the summary dictionary has correct category counts.
        """
        result = self.portfolio_coverage_tvp.get_portfolio_coverage_with_alignment(
            self.data_with_classification,
            PortfolioAggregationMethod.WATS
        )

        summary = result['summary']

        # Should have 2 companies In Transition (A and C)
        self.assertEqual(summary['in_transition']['count'], 2)

        # Should have 2 companies Assessed (B and D) - WB2°C is now Assessed for FINZ
        self.assertEqual(summary['assessed']['count'], 2)

        # Total should be 4
        self.assertEqual(summary['total']['count'], 4)


if __name__ == "__main__":
    test = TestPortfolioCoverageTVP()
    test.setUp()
    test.test_coverage()
