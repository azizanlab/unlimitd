import pickle
import vision_unlimtd

with open("logs_final/shapenet_fim.pickle", "rb") as handle:
    output = pickle.load(handle)

seed = output["seed"]
intermediate_params = output["intermediate_params"]
intermediate_batch_stats = output["intermediate_batch_stats"]
P1 = output["proj"]
intermediate_mean = output["intermediate_mean"]

post_n_epochs = output["post_n_epochs"]
post_n_tasks = output["post_n_tasks"]
post_K = output["post_K"]

data_noise = output["data_noise"]
maddox_noise = output["maddox_noise"]

_, _, post_state, _, post_losses, post_evals = vision_unlimtd.vision_unlimtd_lowdim_cov(seed, intermediate_params, intermediate_batch_stats, post_n_epochs, post_n_tasks, post_K, data_noise, maddox_noise, 0.001, 100, P1, intermediate_mean)

output["post_losses"]=post_losses
output["trained_mean"]=post_state.mean
output["trained_params"]=post_state.params
output["trained_batch_stats"]=post_state.batch_stats
output["trained_scale"]=post_state.scale
output["post_evals"]=post_evals

print("Saving")
with open("logs_final/shapenet_fim.pickle", "wb") as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Ended...")