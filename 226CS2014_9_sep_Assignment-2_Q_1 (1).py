import numpy as np


def calculate_gaussian(pt_x, mu_vec, sigma_mat):
    d_size = len(mu_vec)
    diff_x_mu = pt_x - mu_vec

    det_sigma = np.linalg.det(sigma_mat)

    inv_sigma = np.linalg.inv(sigma_mat)

    norm_const = 1 / (
        (2 * np.pi) ** (d_size / 2) * np.sqrt(det_sigma)
    )

    exp_pow = -0.5 * diff_x_mu.T @ inv_sigma @ diff_x_mu

    prob_density = norm_const * np.exp(exp_pow)

    return prob_density


def classify_bayesian(test_pt, means_arr, covs_arr, priors_arr):

    post_probs = []

    for c_id in range(len(means_arr)):

        likelihood_val = calculate_gaussian(
            test_pt,
            means_arr[c_id],
            covs_arr[c_id]
        )

        post_val = likelihood_val * priors_arr[c_id]

        post_probs.append(post_val)

    post_probs_arr = np.array(post_probs)

    pred_cls = np.argmax(post_probs_arr)

    return pred_cls, post_probs_arr


x_test = np.array([4.0, 4.0])

list_of_means = [
    np.array([2.0, 2.0]),
    np.array([6.0, 6.0])
]

list_of_priors = [
    0.5,
    0.5
]

std_dev = 1.0

cov_ident = std_dev**2 * np.eye(2)

cov_list_1 = [
    cov_ident,
    cov_ident
]

pred_1, probs_1 = classify_bayesian(
    x_test,
    list_of_means,
    cov_list_1,
    list_of_priors
)

print("\nCASE 1: Covariance = sigma^2 I")
print("--------------------------------")

print("Class 1 probability:", probs_1[0])
print("Class 2 probability:", probs_1[1])
print("Predicted Class:", pred_1 + 1)


cov_equal = np.array([
    [2.0, 0.5],
    [0.5, 2.0]
])

cov_list_2 = [
    cov_equal,
    cov_equal
]

pred_2, probs_2 = classify_bayesian(
    x_test,
    list_of_means,
    cov_list_2,
    list_of_priors
)

print("\nCASE 2: Equal Covariance")
print("--------------------------------")

print("Class 1 probability:", probs_2[0])
print("Class 2 probability:", probs_2[1])

print("Predicted Class:", pred_2 + 1)


cov_cls1 = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

cov_cls2 = np.array([
    [3.0, 0.5],
    [0.5, 2.0]
])

cov_list_3 = [
    cov_cls1,
    cov_cls2
]

pred_3, probs_3 = classify_bayesian(
    x_test,
    list_of_means,
    cov_list_3,
    list_of_priors
)

print("\nCASE 3: Unequal Covariance")
print("--------------------------------")

print("Class 1 probability:", probs_3[0])
print("Class 2 probability:", probs_3[1])

print("Predicted Class:", pred_3 + 1)