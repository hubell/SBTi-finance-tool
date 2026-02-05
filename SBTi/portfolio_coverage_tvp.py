from typing import Type, Optional, Dict, Any
import pandas as pd
from SBTi.configs import PortfolioCoverageTVPConfig
from SBTi.portfolio_aggregation import PortfolioAggregation, PortfolioAggregationMethod
from SBTi.interfaces import FINZAlignmentCategory


class PortfolioCoverageTVP(PortfolioAggregation):
    """
    Lookup the companies in the given portfolio and determine whether they have a SBTi approved target.

    :param config: A class defining the constants that are used throughout this class. This parameter is only required
                    if you'd like to overwrite a constant. This can be done by extending the PortfolioCoverageTVPConfig
                    class and overwriting one of the parameters.
    """

    def __init__(
        self, config: Type[PortfolioCoverageTVPConfig] = PortfolioCoverageTVPConfig
    ):
        super().__init__(config)
        self.c: Type[PortfolioCoverageTVPConfig] = config

    def get_portfolio_coverage(
        self,
        company_data: pd.DataFrame,
        portfolio_aggregation_method: PortfolioAggregationMethod,
    ) -> Optional[float]:
        """
        Get the TVP portfolio coverage (i.e. what part of the portfolio has a SBTi validated target).

        :param company_data: The company as it is returned from the data provider's get_company_data call.
        :param portfolio_aggregation_method: PortfolioAggregationMethod: The aggregation method to use
        :return: The aggregated score
        """
        company_data[self.c.OUTPUT_TARGET_STATUS] = company_data.apply(
            lambda row: 100 if row[self.c.COLS.SBTI_VALIDATED] else 0, axis=1
        )

        return self._calculate_aggregate_score(
            company_data, self.c.OUTPUT_TARGET_STATUS, portfolio_aggregation_method
        ).sum()

    def get_finz_portfolio_coverage(
        self,
        company_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Get FINZ climate alignment assessment for a portfolio.

        Per FINZ Standard Table 4.2, companies are categorized as:
        - "In Transition": Companies with 1.5°C validated SBTi targets
        - "Assessed": All other companies (including WB2°C, 2°C, or no target)

        Coverage is calculated on a simple per-entity basis (no weighting):
        - $ coverage: sum of investment values / total portfolio value
        - # coverage: count of companies / total company count

        :param company_data: DataFrame with columns:
            - company_name, company_id, investment_value (required)
            - sbti_validated (required for validation status)
            - target_classification (optional, for granular detail)
        :return: Dictionary containing:
            - company_data: DataFrame with finz_category column added
            - coverage: Coverage metrics ($ and # for each category)
            - summary: High-level summary statistics
        """
        # Make a copy to avoid modifying original
        df = company_data.copy()

        # Ensure required columns exist
        if self.c.COLS.SBTI_VALIDATED not in df.columns:
            raise ValueError(f"Column '{self.c.COLS.SBTI_VALIDATED}' not found in company_data")
        if self.c.COLS.INVESTMENT_VALUE not in df.columns:
            raise ValueError(f"Column '{self.c.COLS.INVESTMENT_VALUE}' not found in company_data")

        # Check for target classification column
        classification_col = None
        for col in ['target_classification', 'Target Classification', 'Near Term Classification']:
            if col in df.columns:
                classification_col = col
                break

        # Determine FINZ category for each company
        # For FINZ: only 1.5°C = "In Transition", everything else = "Assessed"
        def classify_finz(row):
            if not row.get(self.c.COLS.SBTI_VALIDATED, False):
                return FINZAlignmentCategory.ASSESSED.value

            if classification_col and pd.notna(row.get(classification_col)):
                classification = str(row[classification_col])
                # Pure 1.5°C classifications only
                if any(val in classification for val in ['1.5°C', '1.5°C/1.5°C', '1.5']):
                    return FINZAlignmentCategory.IN_TRANSITION.value

            # Has SBT but not 1.5°C = Assessed (for FINZ purposes)
            return FINZAlignmentCategory.ASSESSED.value

        df['finz_category'] = df.apply(classify_finz, axis=1)

        # Calculate totals
        total_value = df[self.c.COLS.INVESTMENT_VALUE].sum()
        total_count = len(df)

        # Calculate coverage for each category (simple sum, no weighting)
        in_transition_mask = df['finz_category'] == FINZAlignmentCategory.IN_TRANSITION.value
        assessed_mask = df['finz_category'] == FINZAlignmentCategory.ASSESSED.value

        in_transition_value = df.loc[in_transition_mask, self.c.COLS.INVESTMENT_VALUE].sum()
        in_transition_count = in_transition_mask.sum()

        assessed_value = df.loc[assessed_mask, self.c.COLS.INVESTMENT_VALUE].sum()
        assessed_count = assessed_mask.sum()

        # Calculate percentages
        in_transition_value_pct = (in_transition_value / total_value * 100) if total_value > 0 else 0
        in_transition_count_pct = (in_transition_count / total_count * 100) if total_count > 0 else 0

        assessed_value_pct = (assessed_value / total_value * 100) if total_value > 0 else 0
        assessed_count_pct = (assessed_count / total_count * 100) if total_count > 0 else 0

        coverage = {
            'in_transition': {
                'value': in_transition_value,
                'value_pct': in_transition_value_pct,
                'count': int(in_transition_count),
                'count_pct': in_transition_count_pct,
            },
            'assessed': {
                'value': assessed_value,
                'value_pct': assessed_value_pct,
                'count': int(assessed_count),
                'count_pct': assessed_count_pct,
            },
            'total': {
                'value': total_value,
                'count': int(total_count),
            }
        }

        summary = {
            'finz_in_transition_value_pct': in_transition_value_pct,
            'finz_in_transition_count_pct': in_transition_count_pct,
            'total_portfolio_value': total_value,
            'total_companies': int(total_count),
        }

        return {
            'company_data': df,
            'coverage': coverage,
            'summary': summary,
        }

    def get_portfolio_coverage_with_alignment(
        self,
        company_data: pd.DataFrame,
        portfolio_aggregation_method: PortfolioAggregationMethod,
    ) -> Dict[str, Any]:
        """
        Get the TVP portfolio coverage with FINZ alignment categorization.

        DEPRECATED: Use get_finz_portfolio_coverage() for FINZ assessments.
        This method is kept for backward compatibility with FINT workflows.

        Returns both FINT coverage (all validated SBT targets) and FINZ coverage
        (only 1.5°C targets, per FINZ "In Transition" criteria).

        :param company_data: The company data with sbti_validated, target_classification,
                             and investment_value columns.
        :param portfolio_aggregation_method: The aggregation method to use
        :return: Dictionary containing:
            - fint_coverage_pct: % of portfolio with any validated SBT target
            - finz_in_transition_pct: % with 1.5°C targets (FINZ "In Transition")
            - company_data: DataFrame with added finz_category column
            - summary: Category breakdown by count and value
        """
        # Make a copy to avoid modifying original
        df = company_data.copy()

        # Ensure required columns exist
        if self.c.COLS.SBTI_VALIDATED not in df.columns:
            raise ValueError(f"Column '{self.c.COLS.SBTI_VALIDATED}' not found in company_data")

        # Check for target classification column
        classification_col = None
        for col in ['target_classification', 'Target Classification', 'Near Term Classification']:
            if col in df.columns:
                classification_col = col
                break

        # Determine FINZ category for each company
        def classify_finz(row):
            if not row.get(self.c.COLS.SBTI_VALIDATED, False):
                return FINZAlignmentCategory.ASSESSED.value

            if classification_col and pd.notna(row.get(classification_col)):
                classification = str(row[classification_col])
                # Pure 1.5°C classifications
                if any(val in classification for val in ['1.5°C', '1.5°C/1.5°C', '1.5']):
                    return FINZAlignmentCategory.IN_TRANSITION.value

            # Has SBT but not 1.5°C = Assessed for FINZ
            return FINZAlignmentCategory.ASSESSED.value

        df['finz_category'] = df.apply(classify_finz, axis=1)

        # Calculate FINT coverage (any validated SBT target)
        df[self.c.OUTPUT_TARGET_STATUS] = df.apply(
            lambda row: 100 if row[self.c.COLS.SBTI_VALIDATED] else 0, axis=1
        )
        fint_coverage = self._calculate_aggregate_score(
            df, self.c.OUTPUT_TARGET_STATUS, portfolio_aggregation_method
        ).sum()

        # Calculate FINZ "In Transition" coverage (only 1.5°C targets)
        df['finz_in_transition_score'] = df.apply(
            lambda row: 100 if row['finz_category'] == FINZAlignmentCategory.IN_TRANSITION.value else 0,
            axis=1
        )
        finz_in_transition = self._calculate_aggregate_score(
            df, 'finz_in_transition_score', portfolio_aggregation_method
        ).sum()

        # Calculate summary statistics
        total_value = df[self.c.COLS.INVESTMENT_VALUE].sum() if self.c.COLS.INVESTMENT_VALUE in df.columns else 0

        summary = {
            'in_transition': {
                'count': len(df[df['finz_category'] == FINZAlignmentCategory.IN_TRANSITION.value]),
                'value': df[df['finz_category'] == FINZAlignmentCategory.IN_TRANSITION.value][self.c.COLS.INVESTMENT_VALUE].sum() if self.c.COLS.INVESTMENT_VALUE in df.columns else 0,
            },
            'assessed': {
                'count': len(df[df['finz_category'] == FINZAlignmentCategory.ASSESSED.value]),
                'value': df[df['finz_category'] == FINZAlignmentCategory.ASSESSED.value][self.c.COLS.INVESTMENT_VALUE].sum() if self.c.COLS.INVESTMENT_VALUE in df.columns else 0,
            },
            'total': {
                'count': len(df),
                'value': total_value,
            }
        }

        # Clean up temporary column
        df = df.drop(columns=['finz_in_transition_score'])

        return {
            'fint_coverage_pct': fint_coverage,
            'finz_in_transition_pct': finz_in_transition,
            'company_data': df,
            'summary': summary,
        }
