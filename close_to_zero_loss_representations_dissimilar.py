"""

Example showing that having close to zero loss is not enough 
to make functions close to identifiable. 


"""

import matplotlib.pyplot as plt
import numpy as np 
from numpy.typing import NDArray

from scipy.special import logsumexp

from sklearn.cross_decomposition import CCA



# We consider 3 models which all have close to 0 loss, but where two of them 
# are close to identifiable and the third are far from the others.
# Model 1, is the model called \theta' in the paper
# Model 3, is the model called \theta* in the paper 

# The models have 4 classes (we call these y_0, y_1, y_2 and y_3) and we use 
# representation dimension M = 2.

# We first introduce the models then we compare them.  


# MODEL 1 (\theta')
# Model 1 has a g function, g_1, such that g_1(y_0) = (a_0, 0), 
# g_1(y_3) = (0, b_3) and g_1(y_1), g_1(y_2) are between the two 
# with \pi/16 radians between each vector. 
# We assume model 1 has an f function, f_1, such that ||f_1(x)|| >= 1 for all x, 
# and each f_1(x) is within \pi/128 radians of the correct label. 
f_deviating_radians = np.pi/128
print(np.cos(f_deviating_radians)) # 0.9951847266721969
# That is, cos(angle(f_1(x), g_1(y))) > 0.995 if y is the correct label for x. 

g_1_as = np.array([1, np.cos(np.pi/16), np.cos(2*np.pi/16), np.cos(3*np.pi/16)])
g_1_bs = np.array([0, np.sin(np.pi/16), np.sin(2*np.pi/16), np.sin(3*np.pi/16)])

plt.figure(figsize=(5.6,5.3))
plt.scatter(g_1_as, g_1_bs)
plt.ylim(-1.1, 1.1)
plt.xlim(-1.1, 1.1)
plt.grid()
plt.show()

# To find out what ||g_1(y)|| should be, we consider a "worst case" x. 
# That is, we consider an x which will have p(y|x, S) as small as possible 
# for the correct label, y, and set ||g_1(y)|| such that we still have
# p(y|x, S) > 0.99. 
# We take an x, x_w, with correct label y_1. We assume that f_1(x_w) is 
# between g_1(y_1) and g_1(y_2) with
# angle(f_1(x_w), g_1(y_1)) = \pi/128 
# then we have: 
# angle(f_1(x_w), g_1(y_0)) = \pi/16 + \pi/128 
# angle(f_1(x_w), g_1(y_2)) = \pi/16 - \pi/128
# angle(f_1(x_w), g_1(y_3)) = 2\pi/16 - \pi/128

# Since f_1(x_w)^Tg_1(y) = cos(f_1(x_w), g_1(y)) * || f_1(x_w) || * || g_1(y) ||, we have 
#
# p(y_1 | x_w, S) =  exp(cos(f_1(x_w), g_1(y_1)) * || f_1(x_w) || * || g_1(y_1) ||) /
#                       exp(cos(f_1(x_w), g_1(y_1)) * || f_1(x_w) || * || g_1(y_1) ||)
#                       + exp(cos(f_1(x_w), g_1(y_0)) * || f_1(x_w) || * || g_1(y_0) ||)
#                       + exp(cos(f_1(x_w), g_1(y_2)) * || f_1(x_w) || * || g_1(y_2) ||)
#                       + exp(cos(f_1(x_w), g_1(y_3)) * || f_1(x_w) || * || g_1(y_3) ||)
#                 =  exp(cos(\pi/128) * || f_1(x_w) || * || g_1(y_0) ||) /
#                       exp(cos(\pi/128) * || f_1(x_w) || * || g_1(y_1) ||)
#                       + exp(cos(\pi/16 + \pi/128) * || f_1(x_w) || * || g_1(y_0) ||)
#                       + exp(cos(\pi/16 - \pi/128) * || f_1(x_w) || * || g_1(y_2) ||)
#                       + exp(cos(2\pi/16 - \pi/128) * || f_1(x_w) || * || g_1(y_3) ||)

# We now consider the function, h_1(v), where 
# h_1(v) = exp(cos(\pi/128) * v) /
#                       exp(cos(\pi/128) * v)
#                       + exp(cos(\pi/16 + \pi/128) * v)
#                       + exp(cos(\pi/16 - \pi/128) * v)
#                       + exp(cos(2\pi/16 - \pi/128) * v)  

m1_f_angles = np.array([np.pi/128, np.pi/16 + np.pi/128, np.pi/16 - np.pi/128, 2*np.pi/16 + np.pi/128]) 
m1_f_angles = np.expand_dims(m1_f_angles, 0)

def h_1(v):
    """ Get the likelihood of model 1 
        as function of product of the representation lengths"""
    return np.exp(np.cos(np.pi/128) * v) / (np.exp(np.cos(np.pi/128) * v) 
                                            + np.exp(np.cos(np.pi/16 + np.pi/128) * v) 
                                            + np.exp(np.cos(np.pi/16 - np.pi/128) * v) 
                                            + np.exp(np.cos(2*np.pi/16 + np.pi/128) * v))

def h_1_logvec(v):
    """ Get the log-likelihood of model 1
        as function of product of the representation lengths"""
    v_norm = np.expand_dims(v, 1)
    return (np.cos(np.pi/128) * v) - (logsumexp(np.cos(m1_f_angles) * v_norm, axis = 1))

# Since lengths of vectors are always non-negative, we consider h(v) with v \in [0, 100]
v_vals = np.linspace(0, 1000, num=1000)

plt.plot(v_vals, h_1(v_vals))
plt.grid()
plt.show()

plt.plot(v_vals, h_1_logvec(v_vals))
plt.grid()
plt.show()


# Thus we see that we can bring p(y_1 | x_w, S) arbitrarily close to 1, by increasing v. 
# For example, since we assumed ||f_1(x)|| >= 1 for all x, if we set || g_1(y) || = 600, 
# we get 
print(h_1(600)) # 0.9998248185618481
# p(y_1 | x_w, S) > 0.99, even in the worst case. 
# Therefore the loss, which is an expectation of negative log-likelihoods, will be smaller 
# than -log(0.99) < 0.011
print(-np.log(0.99)) # 0.01005033585350145

# In the example in the paper we set || g_1(y) || = 1100
# So we get a "worst case" log-likelihood of 
print(h_1_logvec(np.array([1100]))) # -1.29024102e-07


# MODEL 2 (Not in the paper)
# Model 2 has a g function, g_2, such that g_2(y_0) = (a_0, 0), g_2(y_1) = (0, b_1), 
# g_2(y_2) = (-a_0, 0) and g_2(0, -b_1).  
# So there are \pi/2 radians between each g vector. 

g_2_as = np.array([1, 0, -1, 0])
g_2_bs = np.array([0, 1, 0, -1])

plt.figure(figsize=(5.6,5.3))
plt.scatter(g_2_as, g_2_bs)
plt.ylim(-1.1, 1.1)
plt.xlim(-1.1, 1.1)
plt.grid()
plt.show()

# We assume model 2 has an f function, f_2, such that ||f_2(x)|| >= 1 for all x, 
# and each f_2(x) is within \pi/32 radians of the correct label. 
# That is, cos(angle(f_2(x), g_2(y))) > 0.995 if y is the correct label for x. 

# To find out what ||g_2(y)|| should be, we again consider a "worst case" x. 
# That is, we consider an x which will have p(y|x, S) as small as possible for 
# the correct label, y, and set ||g_2(y)|| such that we still have p(y|x, S) > 0.99. 
# We take an x, x_w, with correct label y_1. We assume that f_2(x_w) is between 
# g_2(y_1) and g_2(y_2) with
# angle(f_2(x_w), g_2(y_1)) = \pi/32 
# then we have: 
# angle(f_1(x_w), g_1(y_0)) = \pi/2 + \pi/32 
# angle(f_1(x_w), g_1(y_2)) = \pi/2 - \pi/32
# angle(f_1(x_w), g_1(y_3)) = \pi - \pi/32

# We consider the function h_2(v) where 
# h_2(v) = exp(cos(\pi/32) * v) /
#                       exp(cos(\pi/32) * v)
#                       + exp(cos(\pi/2 + \pi/32) * v)
#                       + exp(cos(\pi/2 - \pi/32) * v)
#                       + exp(cos(\pi - \pi/32) * v)  

def h_2(v):
    """ Get the likelihood of model 2 
        as function of product of the representation lengths"""
    return np.exp(np.cos(np.pi/16) * v) / (np.exp(np.cos(np.pi/16) * v) 
                                            + np.exp(np.cos(np.pi/2 + np.pi/16) * v) 
                                            + np.exp(np.cos(np.pi/2 - np.pi/16) * v) 
                                            + np.exp(np.cos(np.pi + np.pi/16) * v))

v_vals = np.linspace(0, 100, num=1000)

plt.plot(v_vals, h_2(v_vals))
plt.grid()
plt.show()

# Thus we again see that we can bring p(y_1 | x_w, S) arbitrarily close to 1, by increasing v. 
# Since we assumed ||f_2(x)|| >= 1 for all x, we can set || g_2(y) || = 20, and get
print(h_2(20)) # 0.9999999835627914
# p(y_1 | x_w, S) > 0.99, even in the worst case. 
# Therefore the loss, which is an expectation of negative log-likelihoods, will be smaller 
# than -log(0.99) < 0.011
print(-np.log(0.99)) # 0.01005033585350145


# MODEL 3 (\theta*)
# Model 3 has a g function, g_3, such that || g_3(y) || = 20, there is \pi/2 - \pi/6 radians 
# between g_3(y_0) and g_3(y_1), \pi/2 + \pi/32 radians between g_3(y_1) and g_3(y_2)
# and \pi/2 radians between g_3(y_2) and g_3(y_3). 

# So g_3(y_0) = (20, 0), g_3(y_2) = (-20, 0), g_3(y_3) = (0, -20) and  
# g_3(y_0) = (20*cos(\pi/2 - \pi/6), 20*sin(\pi/2 - \pi/6)) 
print(20*np.cos(np.pi/2 - np.pi/6))
print(20*np.sin(np.pi/2 - np.pi/6))
# = (9.999999999999998, 17.320508075688775)

# scaled by 1/20
g_3_as = np.array([1, np.cos(np.pi/2 - np.pi/6), -1, 0])
g_3_bs = np.array([0, np.sin(np.pi/2 - np.pi/6), 0, -1])

plt.figure(figsize=(5.6,5.3))
plt.scatter(g_3_as, g_3_bs)
plt.ylim(-1.1, 1.1)
plt.xlim(-1.1, 1.1)
plt.grid()
plt.show()

# We assume model 3 has an f function, f_3, such that ||f_3(x)|| >= 1 for all x, 
# and each f_3(x) is within \pi/32 radians of the correct label. 
# That is, cos(angle(f_3(x), g_3(y))) > 0.995 if y is the correct label for x. 



# CONSTRUCTION of the embedding functions, f(x) values 

# We first represent the x's belonging to each label, y, by the angle between f(x) 
# and g(y) for the correct label y together with the length of f(x), || f(x) ||. 
# For each label, we draw 1000 values from a uniform distribution U[-\pi/32, \pi/32] 
# to represent the angle, and draw 1000 values from a standard normal distribution, 
# z from N(0, 1), and transform it with |z| + 1, to get the lengths of the vectors. 
rng = np.random.default_rng()
x_angles_m1 = rng.uniform(-np.pi/128, np.pi/128, (4, 1000))
x_angles_m23 = x_angles_m1*16
fx_lengths = np.abs(rng.standard_normal((4, 1000))) + 1

# We then represent the g(y)s  for the three models using their angles to (1, 0) 
g_1_angles = np.expand_dims(np.array([0, np.pi/16, 2*np.pi/16, 3*np.pi/16]), 1)
g_2_angles = np.expand_dims(np.array([0, np.pi/2, np.pi, 3*np.pi/2]), 1)
g_3_angles = np.expand_dims(np.array([0, np.pi/2 - np.pi/6, np.pi, 3*np.pi/2]), 1) 

# We get the g(y) values
def get_gy_from_rad_and_length(
    g_rad: NDArray[float], g_length: float):
    gy_a = g_length*np.cos(g_rad)
    gy_b = g_length*np.sin(g_rad)
    gy = np.concatenate((np.expand_dims(gy_a, 2), np.expand_dims(gy_b, 2)), 2)

    return gy


m1_gy = get_gy_from_rad_and_length(g_1_angles, 1100)

m2_gy = get_gy_from_rad_and_length(g_2_angles, 25)

m3_gy = get_gy_from_rad_and_length(g_3_angles, 100)


# We get the final f(x) values
def get_fx_from_rad_from_g_and_length(
    f_rad_from_g: NDArray[float], g_angle: NDArray[float], f_length: NDArray[float]):
    f_angle = (g_angle + f_rad_from_g) % (2*np.pi)
    fx_a = f_length*np.cos(f_angle)
    fx_b = f_length*np.sin(f_angle)
    fx = np.concatenate((np.expand_dims(fx_a, 2), np.expand_dims(fx_b, 2)), 2)

    return fx 

# Plot f(x) for model 1
m1_fx = get_fx_from_rad_from_g_and_length(x_angles_m1, g_1_angles, fx_lengths)
m1_all_fx = np.concatenate(m1_fx, axis = 0)

plt.figure(figsize=(4.05,4))
plt.scatter(m1_fx[0, :, 0], m1_fx[0, :, 1], c='#a6cee3', alpha=0.5, s=5, edgecolors='none')
plt.scatter(m1_fx[1, :, 0], m1_fx[1, :, 1], c='#1f78b4', alpha=0.5, s=5, edgecolors='none')
plt.scatter(m1_fx[2, :, 0], m1_fx[2, :, 1], c='#b2df8a', alpha=0.5, s=5, edgecolors='none')
plt.scatter(m1_fx[3, :, 0], m1_fx[3, :, 1], c='#33a02c', alpha=0.5, s=5, edgecolors='none')
plt.ylim(-4.6, 4.6)
plt.xlim(-4.6, 4.6)
plt.grid()
plt.tight_layout()
plt.savefig('model_1_fx_pi_over_16.png', dpi=300)
plt.clf()
#plt.show()


# Plot f(x) for model 2
m2_fx = get_fx_from_rad_from_g_and_length(x_angles_m23, g_2_angles, fx_lengths)
m2_all_fx = np.concatenate(m2_fx, axis = 0)

plt.figure(figsize=(4.05,4))
plt.scatter(m2_fx[0, :, 0], m2_fx[0, :, 1], c='#a6cee3', alpha=0.5, s=5, edgecolors='none')
plt.scatter(m2_fx[1, :, 0], m2_fx[1, :, 1], c='#1f78b4', alpha=0.5, s=5, edgecolors='none')
plt.scatter(m2_fx[2, :, 0], m2_fx[2, :, 1], c='#b2df8a', alpha=0.5, s=5, edgecolors='none')
plt.scatter(m2_fx[3, :, 0], m2_fx[3, :, 1], c='#33a02c', alpha=0.5, s=5, edgecolors='none')
plt.ylim(-4.6, 4.6)
plt.xlim(-4.6, 4.6)
plt.grid()
plt.tight_layout()
plt.savefig('model_2_fx_pi_over_2.png', dpi=300)
plt.clf()


# Plot f(x) for model 3
m3_fx = get_fx_from_rad_from_g_and_length(x_angles_m23, g_3_angles, fx_lengths)
m3_all_fx = np.concatenate(m3_fx, axis = 0)

plt.figure(figsize=(4.05,4))
plt.scatter(m3_fx[0, :, 0], m3_fx[0, :, 1], c='#a6cee3', alpha=0.5, s=5, edgecolors='none')
plt.scatter(m3_fx[1, :, 0], m3_fx[1, :, 1], c='#1f78b4', alpha=0.5, s=5, edgecolors='none')
plt.scatter(m3_fx[2, :, 0], m3_fx[2, :, 1], c='#b2df8a', alpha=0.5, s=5, edgecolors='none')
plt.scatter(m3_fx[3, :, 0], m3_fx[3, :, 1], c='#33a02c', alpha=0.5, s=5, edgecolors='none')
plt.ylim(-4.6, 4.6)
plt.xlim(-4.6, 4.6)
plt.grid()
plt.tight_layout()
plt.savefig('model_3_fx_pi_over_2_deviate.png', dpi=300)
plt.clf()
#plt.show()


# We calculate the loss for these three models 
# p(y|x, S) = exp(f(x)^Tg(y)) / (\sum_{y_i \in S} exp(f(x)^Tg(y_i)))
# loss = mean(-log(p(y|x, S))) of all x for correct labels, y. 
def dot(fx: NDArray[float], gy: NDArray[float]):
    """ Dot product batched """
    return np.matmul(np.expand_dims(fx, 1), np.expand_dims(gy, 2))


def log_likelihood(fx: NDArray[float], gy: NDArray[float], all_gys: NDArray[float]):
    fg_dot = dot(fx, gy)

    normalisation = np.zeros((*fg_dot.shape, all_gys.shape[0]))

    for i, current_target in enumerate(all_gys):
        n = dot(fx, current_target)
        normalisation[:, :, :, i] = n 
    
    normalisation = logsumexp(normalisation, axis = 3, keepdims = False)

    return fg_dot - normalisation

# Log likelihoods for model 1
m1_y0_log_p = log_likelihood(m1_fx[0, :], m1_gy[0], m1_gy)
m1_y1_log_p = log_likelihood(m1_fx[1, :], m1_gy[1], m1_gy)
m1_y2_log_p = log_likelihood(m1_fx[2, :], m1_gy[2], m1_gy)
m1_y3_log_p = log_likelihood(m1_fx[3, :], m1_gy[3], m1_gy)

# NLL for model 1
m1_nll = np.mean(
    [np.mean(-m1_y0_log_p), np.mean(-m1_y1_log_p), np.mean(-m1_y2_log_p), np.mean(-m1_y3_log_p)])


# Log likelihoods for model 2
m2_y0_log_p = log_likelihood(m2_fx[0, :], m2_gy[0], m2_gy)
m2_y1_log_p = log_likelihood(m2_fx[1, :], m2_gy[1], m2_gy)
m2_y2_log_p = log_likelihood(m2_fx[2, :], m2_gy[2], m2_gy)
m2_y3_log_p = log_likelihood(m2_fx[3, :], m2_gy[3], m2_gy)

# NLL for model 1
m2_nll = np.mean(
    [np.mean(-m2_y0_log_p), np.mean(-m2_y1_log_p), np.mean(-m2_y2_log_p), np.mean(-m2_y3_log_p)])


# Log likelihoods for model 3
m3_y0_log_p = log_likelihood(m3_fx[0, :], m3_gy[0], m3_gy)
m3_y1_log_p = log_likelihood(m3_fx[1, :], m3_gy[1], m3_gy)
m3_y2_log_p = log_likelihood(m3_fx[2, :], m3_gy[2], m3_gy)
m3_y3_log_p = log_likelihood(m3_fx[3, :], m3_gy[3], m3_gy)

# NLL for model 1
m3_nll = np.mean(
    [np.mean(-m3_y0_log_p), np.mean(-m3_y1_log_p), np.mean(-m3_y2_log_p), np.mean(-m3_y3_log_p)])

print(f'NLL for models. m1: {m1_nll}, m2: {m2_nll}, m3: {m3_nll}')
# NLL for models. m1: 8.805713491710776e-10, m2: 5.417607545687986e-09, m3: 6.15809867099415e-10


# Max difference across all labels 
m1_2_y0_diff_max = np.max(np.abs(m1_y0_log_p - m2_y0_log_p))
m1_2_y1_diff_max = np.max(np.abs(m1_y1_log_p - m2_y1_log_p))
m1_2_y2_diff_max = np.max(np.abs(m1_y2_log_p - m2_y2_log_p))
m1_2_y3_diff_max = np.max(np.abs(m1_y3_log_p - m2_y3_log_p))

m1_2_diff_max = np.max([m1_2_y0_diff_max, m1_2_y1_diff_max, m1_2_y2_diff_max, m1_2_y3_diff_max]) 
print(f'Max difference m1 vs m2 in loss: {m1_2_diff_max}')
# Max difference m1 vs m2 in loss: 1.0775555630004874e-06

m1_3_y0_diff_max = np.max(np.abs(m1_y0_log_p - m3_y0_log_p))
m1_3_y1_diff_max = np.max(np.abs(m1_y1_log_p - m3_y1_log_p))
m1_3_y2_diff_max = np.max(np.abs(m1_y2_log_p - m3_y2_log_p))
m1_3_y3_diff_max = np.max(np.abs(m1_y3_log_p - m3_y3_log_p))

m1_2_diff_max = np.max([m1_2_y0_diff_max, m1_2_y1_diff_max, m1_2_y2_diff_max, m1_2_y3_diff_max]) 
print(f'Max difference m1 vs m3 in loss: {m1_2_diff_max}')
# Max difference m1 vs m3 in loss: 1.0775555630004874e-06


# Getting CCA scores between the models

# Model 1 vs model 2
n_components = 2
cca = CCA(n_components=n_components, max_iter=1000)
cca.fit(m1_all_fx, m2_all_fx)
score = cca.score(m1_all_fx, m2_all_fx)
print(f'Model 1 vs model 2, f(x) CCA score: {score}') 
# Model 1 vs model 2, f(x) CCA score: -0.0941835554293351


# Mean of the CCA correlations
X_c, Y_c = cca.transform(m1_all_fx, m2_all_fx)
corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
mean_corr = np.mean(corrs)
print(f'Model 1 vs model 2, f(x) mean correlation: {mean_corr}') 
# Model 1 vs model 2, f(x) mean correlation: 0.4795881863150534


cca = CCA(n_components=n_components, max_iter=1000)
cca.fit(np.squeeze(m1_gy), np.squeeze(m2_gy))
score = cca.score(np.squeeze(m1_gy), np.squeeze(m2_gy))
print(f'Model 1 vs model 2, g(y) CCA score: {score}') 
# Model 1 vs model 2, g(y) CCA score: 0.47529094795895865

# Mean of the CCA correlations
X_c, Y_c = cca.transform(np.squeeze(m1_gy), np.squeeze(m2_gy))
corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
mean_corr = np.mean(corrs)
print(f'Model 1 vs model 2, g(y) mean correlation: {mean_corr}') 
# Model 1 vs model 2, g(y) mean correlation: 0.9480771677930537


# Model 1 vs model 3
cca = CCA(n_components=2, max_iter=1000)
cca.fit(m1_all_fx, m3_all_fx)
score = cca.score(m1_all_fx, m3_all_fx)
print(f'Model 1 vs model 3, CCA score: {score}') 
# Model 1 vs model 3, CCA score: -0.12202263449871598


# Mean of the CCA correlations
X_c, Y_c = cca.transform(m1_all_fx, m3_all_fx)
# np.corrcoef makes correlation coefficient matrix. [[xx, xy],[yx, yy]]
corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
mean_corr = np.mean(corrs)
print(f'Model 1 vs model 3, f(x) mean correlation: {mean_corr}') 
# Model 1 vs model 3, f(x) mean correlation: 0.43265615177600025



# Check how much random noise changes the score and mean CCA correlation
# Model 1 vs model 1 + noise
cca = CCA(n_components=2, max_iter=1000)
noise_seed = 0
current_noise_level = 2

m1_var_dims = np.var(m1_all_fx, axis = 0)

noise_rng = np.random.default_rng(seed=noise_seed)
noise_dim1 = rng.normal(0,np.sqrt(m1_var_dims[0]*current_noise_level), m1_all_fx.shape[0])
noise_dim2 = rng.normal(0,np.sqrt(m1_var_dims[1]*current_noise_level), m1_all_fx.shape[0])

m1_fx_noised = np.copy(m1_all_fx)
m1_fx_noised[:, 0] = m1_fx_noised[:, 0] + noise_dim1
m1_fx_noised[:, 1] = m1_fx_noised[:, 1] + noise_dim2

cca.fit(m1_all_fx, m1_fx_noised)
score = cca.score(m1_all_fx, m1_fx_noised)
print(f'Model 1 vs model 1 plus noise, CCA score: {score}') 
# noise level 2: Model 1 vs model 1 plus noise, CCA score: 0.1470151570706495
# noise level 3: Model 1 vs model 1 plus noise, CCA score: 0.008641259922772926
# noise level 4: Model 1 vs model 1 plus noise, CCA score: -0.10776185733517363


# Mean of the CCA correlations
X_c, Y_c = cca.transform(m1_all_fx, m1_fx_noised)
# np.corrcoef makes correlation coefficient matrix. [[xx, xy],[yx, yy]]
corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
mean_corr = np.mean(corrs)
print(f'Model 1 vs model 1 plus noise, f(x) mean correlation: {mean_corr}') 
# noise level 2: Model 1 vs model 1 plus noise, f(x) mean correlation: 0.5655786204087068
# noise level 3: Model 1 vs model 1 plus noise, f(x) mean correlation: 0.5035250453313113
# noise level 4: Model 1 vs model 1 plus noise, f(x) mean correlation: 0.44089715694684717



# Model 2 vs model 3
cca = CCA(n_components=2, max_iter=1000)
cca.fit(m2_all_fx, m3_all_fx)
score = cca.score(m2_all_fx, m3_all_fx)
print(f'Model 2 vs model 3, CCA score: {score}') 
# Model 2 vs model 3, CCA score: 0.9713806001236809


