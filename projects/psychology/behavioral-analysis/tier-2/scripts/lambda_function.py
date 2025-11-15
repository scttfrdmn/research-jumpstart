#!/usr/bin/env python3
"""
Lambda Function for Behavioral Data Analysis

Processes behavioral experiment data from S3:
1. Reads CSV file with trial-level data
2. Calculates statistical measures
3. Performs signal detection theory analysis
4. Fits computational models
5. Stores results in DynamoDB

Environment Variables:
    DYNAMODB_TABLE: Name of DynamoDB table for results
    S3_BUCKET: Name of S3 bucket (optional)
"""

import json
from datetime import datetime
from io import StringIO

import boto3
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# Initialize AWS clients
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")


def lambda_handler(event, context):
    """
    Main Lambda handler function.

    Expected event format (S3 trigger):
    {
        "Records": [{
            "s3": {
                "bucket": {"name": "bucket-name"},
                "object": {"key": "raw/sub001_stroop.csv"}
            }
        }]
    }
    """
    try:
        # Parse S3 event
        if "Records" in event and len(event["Records"]) > 0:
            bucket = event["Records"][0]["s3"]["bucket"]["name"]
            key = event["Records"][0]["s3"]["object"]["key"]
        elif "bucket" in event and "key" in event:
            # Manual invocation
            bucket = event["bucket"]
            key = event["key"]
        else:
            raise ValueError("Invalid event format. Expected S3 trigger or manual invocation.")

        print(f"Processing: s3://{bucket}/{key}")

        # Download and parse CSV
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        csv_data = obj["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(csv_data))

        print(f"Loaded {len(df)} trials")

        # Validate data format
        required_columns = ["participant_id", "trial", "task_type", "rt", "accuracy"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")

        # Extract metadata
        participant_id = df["participant_id"].iloc[0]
        task_type = df["task_type"].iloc[0]

        print(f"Analyzing {participant_id} on {task_type} task")

        # Perform analysis
        results = analyze_behavioral_data(df, participant_id, task_type)

        # Add metadata
        results["timestamp"] = datetime.utcnow().isoformat()
        results["s3_bucket"] = bucket
        results["s3_key"] = key
        results["n_trials"] = len(df)

        # Store in DynamoDB
        table_name = (
            context.function_name if hasattr(context, "function_name") else "BehavioralAnalysis"
        )
        store_results(results, table_name)

        print(f"Successfully processed {participant_id}")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Success",
                    "participant_id": participant_id,
                    "task_type": task_type,
                    "results": results,
                }
            ),
        }

    except Exception as e:
        print(f"Error: {e!s}")
        return {"statusCode": 500, "body": json.dumps({"message": "Error", "error": str(e)})}


def analyze_behavioral_data(df, participant_id, task_type):
    """
    Perform comprehensive behavioral analysis.
    """
    results = {"participant_id": participant_id, "task_type": task_type}

    # Basic statistics
    results["mean_rt"] = float(df["rt"].mean())
    results["median_rt"] = float(df["rt"].median())
    results["std_rt"] = float(df["rt"].std())
    results["min_rt"] = float(df["rt"].min())
    results["max_rt"] = float(df["rt"].max())

    results["accuracy"] = float(df["accuracy"].mean())
    results["n_correct"] = int(df["accuracy"].sum())
    results["n_errors"] = int((1 - df["accuracy"]).sum())

    # RT statistics for correct trials only
    correct_trials = df[df["accuracy"] == 1]
    if len(correct_trials) > 0:
        results["mean_rt_correct"] = float(correct_trials["rt"].mean())
        results["median_rt_correct"] = float(correct_trials["rt"].median())
    else:
        results["mean_rt_correct"] = None
        results["median_rt_correct"] = None

    # Task-specific analyses
    if task_type == "stroop":
        results.update(analyze_stroop(df))
    elif task_type == "decision":
        results.update(analyze_decision(df))
    elif task_type == "learning":
        results.update(analyze_learning(df))

    # Signal detection theory (if applicable)
    if task_type in ["stroop", "decision"]:
        sdt = calculate_signal_detection(df)
        results.update(sdt)

    # Fit drift diffusion model (if applicable)
    if task_type in ["stroop", "decision"]:
        try:
            ddm_params = fit_drift_diffusion_model(df)
            results["model_ddm"] = ddm_params
        except Exception as e:
            print(f"Warning: DDM fitting failed: {e}")
            results["model_ddm"] = None

    return results


def analyze_stroop(df):
    """
    Analyze Stroop task data.
    """
    results = {}

    if "stimulus" not in df.columns:
        return results

    # Calculate performance by condition
    congruent = df[df["stimulus"] == "congruent"]
    incongruent = df[df["stimulus"] == "incongruent"]

    if len(congruent) > 0:
        results["mean_rt_congruent"] = float(congruent["rt"].mean())
        results["accuracy_congruent"] = float(congruent["accuracy"].mean())

    if len(incongruent) > 0:
        results["mean_rt_incongruent"] = float(incongruent["rt"].mean())
        results["accuracy_incongruent"] = float(incongruent["accuracy"].mean())

    # Calculate Stroop effect
    if len(congruent) > 0 and len(incongruent) > 0:
        results["stroop_effect_rt"] = results["mean_rt_incongruent"] - results["mean_rt_congruent"]
        results["stroop_effect_accuracy"] = (
            results["accuracy_congruent"] - results["accuracy_incongruent"]
        )

        # Statistical test
        try:
            t_stat, p_value = stats.ttest_ind(incongruent["rt"].values, congruent["rt"].values)
            results["stroop_t_statistic"] = float(t_stat)
            results["stroop_p_value"] = float(p_value)
        except Exception as e:
            print(f"Warning: Stroop t-test failed: {e}")

    return results


def analyze_decision(df):
    """
    Analyze decision making task data.
    """
    results = {}

    if "stimulus" not in df.columns:
        return results

    # Performance by difficulty
    for difficulty in ["easy", "medium", "hard"]:
        subset = df[df["stimulus"] == difficulty]
        if len(subset) > 0:
            results[f"mean_rt_{difficulty}"] = float(subset["rt"].mean())
            results[f"accuracy_{difficulty}"] = float(subset["accuracy"].mean())

    # Calculate difficulty effects
    easy = df[df["stimulus"] == "easy"]
    hard = df[df["stimulus"] == "hard"]

    if len(easy) > 0 and len(hard) > 0:
        results["difficulty_effect_rt"] = float(hard["rt"].mean() - easy["rt"].mean())
        results["difficulty_effect_accuracy"] = float(
            easy["accuracy"].mean() - hard["accuracy"].mean()
        )

    return results


def analyze_learning(df):
    """
    Analyze reinforcement learning task data.
    """
    results = {}

    # Split into blocks for learning curve
    n_trials = len(df)
    n_blocks = 5
    block_size = n_trials // n_blocks

    learning_curve = []
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else n_trials
        block_data = df.iloc[start:end]
        learning_curve.append(float(block_data["accuracy"].mean()))

    results["learning_curve"] = learning_curve

    # Calculate learning rate (slope of learning curve)
    if len(learning_curve) > 1:
        x = np.arange(len(learning_curve))
        slope, intercept = np.polyfit(x, learning_curve, 1)
        results["learning_rate"] = float(slope)
        results["initial_accuracy"] = float(intercept)
        results["final_accuracy"] = float(learning_curve[-1])

    # Fit Q-learning model
    try:
        q_params = fit_q_learning_model(df)
        results["model_q_learning"] = q_params
    except Exception as e:
        print(f"Warning: Q-learning fit failed: {e}")
        results["model_q_learning"] = None

    return results


def calculate_signal_detection(df):
    """
    Calculate signal detection theory metrics (d-prime and criterion).

    Assumes binary classification task.
    """
    results = {}

    try:
        # Count hits, misses, false alarms, correct rejections
        # This is simplified - adjust based on your task structure
        hits = len(df[(df["accuracy"] == 1) & (df["trial"] % 2 == 0)])
        misses = len(df[(df["accuracy"] == 0) & (df["trial"] % 2 == 0)])
        false_alarms = len(df[(df["accuracy"] == 0) & (df["trial"] % 2 == 1)])
        correct_rejections = len(df[(df["accuracy"] == 1) & (df["trial"] % 2 == 1)])

        # Calculate rates
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.5
        fa_rate = (
            false_alarms / (false_alarms + correct_rejections)
            if (false_alarms + correct_rejections) > 0
            else 0.5
        )

        # Avoid extreme values
        hit_rate = np.clip(hit_rate, 0.01, 0.99)
        fa_rate = np.clip(fa_rate, 0.01, 0.99)

        # Calculate d-prime and criterion
        dprime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
        criterion = -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(fa_rate))

        results["dprime"] = float(dprime)
        results["criterion"] = float(criterion)
        results["hit_rate"] = float(hit_rate)
        results["false_alarm_rate"] = float(fa_rate)

    except Exception as e:
        print(f"Warning: SDT calculation failed: {e}")
        results["dprime"] = None
        results["criterion"] = None

    return results


def fit_drift_diffusion_model(df):
    """
    Fit drift diffusion model (simplified version).

    Parameters:
        v: drift rate (quality of evidence)
        a: boundary separation (speed-accuracy tradeoff)
        t: non-decision time (motor response, encoding)
    """
    try:
        # Extract RT and accuracy
        rt = df["rt"].values / 1000  # Convert to seconds
        accuracy = df["accuracy"].values

        # Simple DDM fitting using method of moments
        # This is a simplified version - for production, use HDDM package

        # Initial parameter guesses
        v0 = 0.5  # drift rate
        a0 = 1.0  # boundary separation
        t0 = 0.3  # non-decision time

        def ddm_likelihood(params):
            v, a, t = params

            # Constraints
            if a <= 0 or t < 0 or t > 1:
                return 1e10

            # Simple likelihood based on RT and accuracy
            # More accurate models would use full DDM equations
            predicted_rt = t + a / (2 * v) if v > 0 else t + 1
            predicted_acc = 1 / (1 + np.exp(-v * a))

            rt_error = np.mean((rt - predicted_rt) ** 2)
            acc_error = (accuracy.mean() - predicted_acc) ** 2

            return rt_error + acc_error * 10

        # Optimize
        result = minimize(
            ddm_likelihood, [v0, a0, t0], method="Nelder-Mead", options={"maxiter": 100}
        )

        if result.success:
            v, a, t = result.x
            return {
                "drift_rate": float(v),
                "boundary_separation": float(a),
                "non_decision_time": float(t),
            }
        else:
            return None

    except Exception as e:
        print(f"Warning: DDM fitting error: {e}")
        return None


def fit_q_learning_model(df):
    """
    Fit Q-learning model to reinforcement learning data.

    Parameters:
        alpha: learning rate
        beta: inverse temperature (exploration vs exploitation)
    """
    try:
        # Extract choices and feedback
        choices = df["response"].values
        accuracy = df["accuracy"].values  # Assuming accuracy represents reward

        def q_learning_likelihood(params):
            alpha, beta = params

            # Constraints
            if alpha < 0 or alpha > 1 or beta < 0:
                return 1e10

            # Initialize Q-values
            q_values = {}
            log_likelihood = 0

            for choice, reward in zip(choices, accuracy):
                # Get Q-value for this choice
                if choice not in q_values:
                    q_values[choice] = 0.5

                # Calculate probability of choice (softmax)
                # Simplified: assuming binary choice
                prob_choice = 1 / (1 + np.exp(-beta * q_values[choice]))

                # Update log-likelihood
                if reward == 1:
                    log_likelihood += np.log(prob_choice + 1e-10)
                else:
                    log_likelihood += np.log(1 - prob_choice + 1e-10)

                # Update Q-value
                prediction_error = reward - q_values[choice]
                q_values[choice] += alpha * prediction_error

            return -log_likelihood

        # Optimize
        result = minimize(
            q_learning_likelihood,
            [0.1, 1.0],  # Initial guesses
            method="Nelder-Mead",
            bounds=[(0, 1), (0, 10)],
            options={"maxiter": 100},
        )

        if result.success:
            alpha, beta = result.x
            return {"learning_rate": float(alpha), "inverse_temperature": float(beta)}
        else:
            return None

    except Exception as e:
        print(f"Warning: Q-learning fitting error: {e}")
        return None


def store_results(results, table_name="BehavioralAnalysis"):
    """
    Store analysis results in DynamoDB.
    """
    try:
        table = dynamodb.Table(table_name)

        # Convert numpy types to Python types
        item = convert_to_dynamodb_types(results)

        # Put item
        table.put_item(Item=item)

        print(f"Stored results in DynamoDB table: {table_name}")

    except Exception as e:
        print(f"Error storing results in DynamoDB: {e}")
        raise


def convert_to_dynamodb_types(obj):
    """
    Recursively convert numpy/pandas types to Python types for DynamoDB.
    """
    if isinstance(obj, dict):
        return {k: convert_to_dynamodb_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dynamodb_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


# For local testing
if __name__ == "__main__":
    # Test with sample event
    test_event = {"bucket": "behavioral-data-test", "key": "raw/sub001_stroop.csv"}

    class MockContext:
        function_name = "BehavioralAnalysis"

    result = lambda_handler(test_event, MockContext())
    print(json.dumps(result, indent=2))
