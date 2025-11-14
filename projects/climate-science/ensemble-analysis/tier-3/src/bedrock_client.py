"""
Amazon Bedrock integration for AI-assisted climate analysis.

This module provides functions to use Amazon Bedrock (Claude 3) for
interpreting climate model results, generating reports, and providing
scientific context.
"""

import boto3
import json
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BedrockClimateAssistant:
    """
    AI assistant for climate analysis using Amazon Bedrock.

    Uses Claude 3 to provide interpretation, context, and report generation
    for climate model ensemble results.
    """

    def __init__(
        self,
        model_id: str = 'anthropic.claude-3-sonnet-20240229-v1:0',
        region: str = 'us-east-1'
    ):
        """
        Initialize Bedrock client.

        Parameters
        ----------
        model_id : str
            Bedrock model ID (default: Claude 3 Sonnet)
        region : str
            AWS region (default: us-east-1)
        """
        self.model_id = model_id
        self.client = boto3.client('bedrock-runtime', region_name=region)
        logger.info(f"Initialized Bedrock client with model {model_id}")

    def _invoke_model(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> str:
        """
        Internal method to invoke Bedrock model.

        Parameters
        ----------
        prompt : str
            Input prompt
        max_tokens : int
            Maximum response length
        temperature : float
            Response randomness (0.0-1.0)

        Returns
        -------
        str
            Model response text
        """
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            response_body = json.loads(response['body'].read())
            text = response_body['content'][0]['text']

            logger.info(f"Generated response ({len(text)} characters)")
            return text

        except Exception as e:
            logger.error(f"Bedrock invocation error: {e}")
            raise

    def interpret_projection(
        self,
        stats: Dict,
        region_name: str,
        scenario: str,
        variable: str = 'temperature'
    ) -> str:
        """
        Interpret climate projection results.

        Parameters
        ----------
        stats : dict
            Statistical summary of ensemble results
        region_name : str
            Name of analysis region (e.g., 'US Southwest')
        scenario : str
            Emissions scenario (e.g., 'SSP2-4.5')
        variable : str
            Climate variable analyzed

        Returns
        -------
        str
            Interpretation text

        Examples
        --------
        >>> stats = {'mean': 2.5, 'std': 0.8, 'n_models': 15}
        >>> interpretation = assistant.interpret_projection(
        ...     stats, 'US Southwest', 'SSP2-4.5'
        ... )
        """
        logger.info("Generating projection interpretation")

        prompt = f"""You are a climate scientist analyzing model ensemble projections.
Please interpret the following climate projection results:

Region: {region_name}
Scenario: {scenario}
Variable: {variable}
Statistics: {json.dumps(stats, indent=2)}

Provide a concise scientific interpretation covering:
1. Magnitude of projected change
2. Level of model agreement (based on standard deviation)
3. Confidence in the projection
4. What this means for the region
5. Key uncertainties

Keep the response factual and scientific, around 200-300 words."""

        response = self._invoke_model(prompt, max_tokens=1000, temperature=0.5)
        return response

    def compare_to_literature(
        self,
        results_summary: str,
        region: str,
        variable: str = 'temperature'
    ) -> str:
        """
        Compare results to published literature.

        Parameters
        ----------
        results_summary : str
            Summary of analysis results
        region : str
            Analysis region
        variable : str
            Climate variable

        Returns
        -------
        str
            Literature comparison

        Examples
        --------
        >>> comparison = assistant.compare_to_literature(
        ...     "We find 2.5°C warming by 2050",
        ...     "US Southwest",
        ...     "temperature"
        ... )
        """
        logger.info("Comparing to literature")

        prompt = f"""You are a climate scientist familiar with IPCC reports and
regional climate projections. Compare these new analysis results to published
literature:

Region: {region}
Variable: {variable}
Results: {results_summary}

Provide:
1. How these results compare to IPCC AR6 findings for this region
2. Agreement/disagreement with other regional studies
3. Potential reasons for any differences
4. Confidence assessment

Keep response around 250 words, cite general sources (e.g., "IPCC AR6",
"regional studies") without specific paper titles."""

        response = self._invoke_model(prompt, max_tokens=1200, temperature=0.6)
        return response

    def generate_methods_section(
        self,
        analysis_config: Dict
    ) -> str:
        """
        Generate methods section text for a paper.

        Parameters
        ----------
        analysis_config : dict
            Dictionary with analysis configuration:
            - models: list of models
            - scenario: scenario name
            - variable: variable analyzed
            - region: region description
            - time_period: analysis period
            - baseline: baseline period

        Returns
        -------
        str
            Methods section text

        Examples
        --------
        >>> config = {
        ...     'models': ['CESM2', 'GFDL-CM4', 'UKESM1-0-LL'],
        ...     'scenario': 'SSP2-4.5',
        ...     'variable': 'Surface air temperature',
        ...     'region': 'US Southwest (31-37°N, 114-109°W)',
        ...     'time_period': '2015-2050',
        ...     'baseline': '1995-2014'
        ... }
        >>> methods = assistant.generate_methods_section(config)
        """
        logger.info("Generating methods section")

        prompt = f"""Generate a methods section for a climate science paper based on this analysis:

Configuration:
{json.dumps(analysis_config, indent=2)}

Write a publication-quality methods section (300-400 words) covering:
1. Data sources (CMIP6 from AWS)
2. Model selection and ensemble composition
3. Spatial domain and temporal coverage
4. Analysis methods (regional averaging, anomaly calculation)
5. Ensemble statistics computed
6. Software/tools used (xarray, Python)

Use past tense, third person, academic style. Follow standard climate science
paper conventions."""

        response = self._invoke_model(prompt, max_tokens=1500, temperature=0.4)
        return response

    def identify_outliers_explanation(
        self,
        outlier_models: List[str],
        ensemble_stats: Dict
    ) -> str:
        """
        Explain why certain models might be outliers.

        Parameters
        ----------
        outlier_models : list of str
            Names of outlier models
        ensemble_stats : dict
            Ensemble statistics

        Returns
        -------
        str
            Explanation of potential outlier causes

        Examples
        --------
        >>> explanation = assistant.identify_outliers_explanation(
        ...     ['UKESM1-0-LL'],
        ...     {'mean': 2.0, 'std': 0.5}
        ... )
        """
        logger.info(f"Generating explanation for {len(outlier_models)} outlier(s)")

        prompt = f"""You are a climate modeling expert. Explain why these CMIP6 models
might be outliers in an ensemble analysis:

Outlier models: {', '.join(outlier_models)}
Ensemble statistics: {json.dumps(ensemble_stats, indent=2)}

Provide:
1. Known characteristics of these model(s) (e.g., climate sensitivity,
   parameterizations)
2. Potential reasons for differing projections
3. Whether outliers should be excluded or retained
4. How to communicate this in a paper

Keep response around 200 words, factual and objective."""

        response = self._invoke_model(prompt, max_tokens=1000, temperature=0.6)
        return response

    def generate_figure_caption(
        self,
        figure_description: str,
        analysis_details: Dict
    ) -> str:
        """
        Generate publication-quality figure caption.

        Parameters
        ----------
        figure_description : str
            Description of what the figure shows
        analysis_details : dict
            Details about the analysis

        Returns
        -------
        str
            Figure caption

        Examples
        --------
        >>> caption = assistant.generate_figure_caption(
        ...     "Time series showing ensemble mean and spread",
        ...     {'models': 15, 'scenario': 'SSP2-4.5', 'region': 'US Southwest'}
        ... )
        """
        logger.info("Generating figure caption")

        prompt = f"""Generate a publication-quality figure caption for a climate science paper:

Figure shows: {figure_description}
Analysis details: {json.dumps(analysis_details, indent=2)}

Write a concise caption (100-150 words) that:
1. Describes what is shown
2. Defines key elements (lines, shading, etc.)
3. Provides context (models, scenario, region)
4. Notes important features to observe

Use standard climate science caption style."""

        response = self._invoke_model(prompt, max_tokens=500, temperature=0.4)
        return response

    def suggest_next_analyses(
        self,
        current_results: str,
        analysis_type: str = 'regional projection'
    ) -> str:
        """
        Suggest follow-up analyses based on current results.

        Parameters
        ----------
        current_results : str
            Summary of current analysis results
        analysis_type : str
            Type of analysis performed

        Returns
        -------
        str
            Suggestions for follow-up analyses

        Examples
        --------
        >>> suggestions = assistant.suggest_next_analyses(
        ...     "Found 2.5°C warming with high model agreement",
        ...     "regional projection"
        ... )
        """
        logger.info("Generating analysis suggestions")

        prompt = f"""You are a climate scientist who just completed this analysis:

Analysis type: {analysis_type}
Results: {current_results}

Suggest 3-5 logical next steps or follow-up analyses that would:
1. Build on these findings
2. Address key uncertainties
3. Add scientific value
4. Be feasible with CMIP6 data

For each suggestion, briefly explain why it would be valuable.
Keep total response around 200-250 words."""

        response = self._invoke_model(prompt, max_tokens=1000, temperature=0.7)
        return response

    def check_bedrock_access(self) -> bool:
        """
        Verify Bedrock access is working.

        Returns
        -------
        bool
            True if access works, False otherwise
        """
        try:
            # Try simple invocation
            test_prompt = "Respond with only: 'Bedrock access verified'"
            response = self._invoke_model(test_prompt, max_tokens=50, temperature=0)
            logger.info("✓ Bedrock access verified")
            return True
        except Exception as e:
            logger.error(f"✗ Bedrock access failed: {e}")
            return False
