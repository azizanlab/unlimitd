import vision_unlimtd
import pickle

# run this once supercloud is back up
# with two GPUs
# expected time: 40h (3+ days)

print("Testing dumping")
with open("logs_final/shapenet_fim.pickle", "wb") as handle:
    pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)

seed = 1655235988902897757

init_params, pre_state, pre_evals, post_state, pre_losses, post_losses, post_evals = vision_unlimtd.vision_unlimtd_identitycov(seed=seed,
                                                                                     pre_n_epochs=5000,
                                                                                     pre_n_tasks=10,
                                                                                     pre_K=15,
                                                                                     post_n_epochs=5000,
                                                                                     post_n_tasks=10,
                                                                                     post_K=15,
                                                                                     data_noise=0, 
                                                                                     maddox_noise=0.01,
                                                                                     meta_lr=0.001,
                                                                                     subspace_dimension=100)

output = {}
output["seed"] = seed
output["pre_n_epochs"]=5000
output["pre_n_tasks"]=10
output["pre_K"]=15
output["post_n_epochs"]=5000
output["post_n_tasks"]=10
output["post_K"]=15
output["data_noise"]=0
output["maddox_noise"]=0.01
output["meta_lr"]=0.001
output["subspace_dimension"]=100
output["pre_losses"]=pre_losses
# output["post_losses"]=post_losses
output["init_params"]=init_params
output["intermediate_params"]=pre_state.params
# output["trained_params"]=post_state.params
output["intermediate_mean"]=pre_state.mean
# output["trained_mean"]=post_state.mean
output["intermediate_batch_stats"]=pre_state.batch_stats
# output["trained_batch_stats"]=post_state.batch_stats
# output["trained_scale"]=post_state.scale
# output["proj"]=post_state.proj
output["pre_evals"] = pre_evals
# output["post_evals"] = post_evals

print("Saving")
with open("logs_final/shapenet_fim.pickle", "wb") as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Ended...")