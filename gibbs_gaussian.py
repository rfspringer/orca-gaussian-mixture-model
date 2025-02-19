import numpy as np
import random
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import math


def sample_assignments(assignments, proportions, means, data, params):
    new_assignments = assignments.copy()

    for i in range(params['n']):
        # calculate posterior probabilities
        posterior_probs = np.zeros(params['K'])
        for k in range(params['K']):
            prior = proportions[k]
            f = lambda x: scipy.stats.norm.logpdf(x, loc=means[k], scale=params['sigma'])
            likelihood = f(data[i]).sum()
            posterior_probs[k] = prior + likelihood

        posterior_probs -= np.max(posterior_probs)
        posterior_probs = np.exp(posterior_probs)

        # normalize
        assignment_probs = posterior_probs / np.sum(posterior_probs)

        # sample
        new_assignment = np.random.choice(params['K'], p=assignment_probs)
        new_assignments[i] = new_assignment
    return new_assignments


def sample_means(new_assignments, means, data, params):
    cluster_means = means.copy()
    new_means = means.copy()

    for k in range(params['K']):
        sigma = params['sigma']
        eta = params['eta']

        # gather data points assigned to component k
        data_indices = np.where(new_assignments == k)[0]
        component_data = data[data_indices]

        # update mean
        if len(component_data) > 0:
            center = np.mean(component_data, axis=0)
            cluster_means[k] = center

        num_assignments = len(component_data)
        mean = ((num_assignments / (sigma * sigma)) / (num_assignments / (sigma * sigma) + 1 / (eta * eta))) * cluster_means[k]
        var = 1 / (num_assignments / (sigma * sigma) + 1 / (eta * eta))

        new_means[k] = np.random.normal(mean, math.sqrt(var))
    return new_means


def sample_proportions(new_assignments, params):
    alpha = np.full(params['K'], params['alpha_0'])
    for k in range(params['K']):
        alpha[k] += np.sum(new_assignments == k)
    new_proportions = np.random.dirichlet(alpha)
    return new_proportions

def held_out_log_likelihood(new_data, means, proportions, params):
    predictive_log_prob = 0
    probs = []

    for i in range(len(new_data)):
        log_local_likelihood = 0
        for k in range(params['K']):
            f = lambda x: scipy.stats.norm.logpdf(x, loc=means[k], scale=params['sigma']).sum()
            log_local_cluster_likelihood = np.log(proportions[k]) + f(new_data[i])
            log_local_likelihood = np.logaddexp(log_local_likelihood, log_local_cluster_likelihood )
        probs.append(log_local_likelihood + log_prior(proportions, means, params))
        predictive_log_prob = np.logaddexp(predictive_log_prob, log_local_likelihood + log_prior(proportions, means, params))
        # predictive_log_prob += log_local_likelihood
    probs = np.array(probs)
    max_val = np.max(probs)
    shifted_arr = probs - max_val
    result = max_val + np.log(np.sum(np.exp(shifted_arr)))
    return result


def log_joint(assignments, proportions, means, data, params):    # log joint, held out log likelihood, then comparing to audio files to see
    log_likelihood = 0
    # log prob of data
    for i in range(params['n']):
        f = lambda x: scipy.stats.norm.logpdf(x, loc=means[assignments[i]], scale=params['sigma'])
        log_likelihood += f(data[i]).sum()

    #log prob of assignments given proportions
    # for each z assigned to a cluster, theta likelihood
    for k in range(params['K']):
        log_likelihood += np.log(proportions[k]) * np.sum(assignments == k)
    return log_likelihood + log_prior(proportions, means, params)


def log_prior(proportions, means, params):
    log_prob = 0
    # log prob of means
    # d zero mean gaussians with var of eta^2
    for k in range(params['K']):
        f = lambda x: scipy.stats.norm.logpdf(x, loc=0, scale=params['eta'])
        log_prob += f(means[k]).sum()

    # log prob of proportions
    alpha = np.full(params['K'], params['alpha_0'])
    log_prob += scipy.stats.dirichlet.logpdf(proportions, alpha)
    return log_prob


def gibbs(data, params):
    # Randomly initialize assignments, means, and proportions
    assignments = np.random.randint(0, params['K'], params['n'])
    means = [random.choice(data) for _ in range(params['K'])]
    proportions = np.random.dirichlet(np.ones(params['K']))
    log_joints = []

    for iteration in range(params['max_iterations']):
        new_assignments = sample_assignments(assignments, proportions, means, data, params)
        new_means = sample_means(new_assignments, means, data, params)
        new_proportions = sample_proportions(new_assignments, params)

        # Update parameters
        assignments = np.array(new_assignments)
        means = np.array(new_means)
        proportions = np.array(new_proportions)
        log_joints.append(log_joint(assignments, proportions, means, data, params))

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: \nAssignments = {assignments}, \nMeans = {means}, \nProportions = {proportions}\n")

    # Scatter plot for data points
    plt.scatter(data[:, 5], data[:, 6], c='blue', label='Data Points')

    # Scatter plot for cluster means
    plt.scatter(means[:, 5], means[:, 6], c='red', marker='x', s=100, label='Cluster Means')

    plt.title('Dataset and Cluster Means')
    plt.xlabel('Feature 3')
    plt.ylabel('Feature 4')
    plt.legend()

    plt.grid()
    plt.show()

    return assignments, means, proportions


def plot_feature_scatter(data, means, feature_idx, ax):
    ax.scatter(data[:, feature_idx], data[:, feature_idx + 1], c='blue', label='Data Points')   # Scatter plot for data points
    ax.scatter(means[:, feature_idx], means[:, feature_idx + 1], c='red', marker='x', s=100, label='Cluster Means')    # Scatter plot for cluster means
    ax.set_xlabel(f'Feature {feature_idx}')
    ax.set_ylabel(f'Feature {feature_idx+1}')
    ax.legend()


def plot_means_across_features(data, means):
    num_features = 6
    fig, axes = plt.subplots(num_features // 2, 1, figsize=(10, 2*num_features))

    for feature_idx in range(0, num_features, 2):
        plot_feature_scatter(data, means, feature_idx, axes[int(feature_idx/2)])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("train_embeddings.csv")
    data = np.array(df)[:, 1:]
    corresponding_rows = np.array(df)[:, 0]

    df_test = pd.read_csv('test_embeddings')
    data_test = np.array(df_test)[:, 1:]

    annotation_rows = np.array(df)[:, 0]
    params = {'sigma': 0.01, 'alpha_0': 1, 'eta': 0.01, 'K': 3, 'n': len(data), 'seed': 34, 'max_iterations': 80,
              'embedding_len': len(data[0])}
    likelihoods = []

    for k in range(3, 4):
        # Define the parameters
        params['K'] = k
        assignments, means, proportions = gibbs(data, params)
        log_likelihood = held_out_log_likelihood(data_test, means, proportions, params)
        likelihoods.append(log_likelihood)
        print("likelihoods: ", likelihoods)
        plot_means_across_features(data, means)

    for cluster in range(params['K']):
        print("examples of cluster ", cluster)
        assigned_indices = np.where(assignments == cluster)[0]
        print(corresponding_rows[assigned_indices])


