import os
import numpy as np

if os.path.isfile("policy_taxi_sarsa_avaliacao.npy") and os.path.isfile(
    "qvalues_taxi_sarsa_avaliacao.npy"
):
    taxi_sarsa_policy = np.load("policy_taxi_sarsa_avaliacao.npy")
    qvalues_taxi_sarsa = np.load("qvalues_taxi_sarsa_avaliacao.npy")

if os.path.isfile("policy_taxi_sarsa_lambda_avaliacao.npy") and os.path.isfile(
    "qvalues_taxi_sarsa_lambda_avaliacao.npy"
):
    taxi_sarsa_policy_lambda = np.load("policy_taxi_sarsa_lambda_avaliacao.npy")
    qvalues_taxi_sarsa_lambda = np.load("qvalues_taxi_sarsa_lambda_avaliacao.npy")
