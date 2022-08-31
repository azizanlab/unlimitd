import pickle
import vision_unlimtd

with open("logs_final/shapenet_fim.pickle", "rb") as handle:
    output = pickle.load(handle)

seed = output["seed"]
intermediate_params = output["intermediate_params"]
intermediate_batch_stats = output["intermediate_batch_stats"]
subspace_dimension = output["subspace_dimension"]

P1 = vision_unlimtd.vision_unlimtd_find_proj(seed, intermediate_params, intermediate_batch_stats, subspace_dimension)

output["proj"] = P1

print("Saving")
with open("logs_final/shapenet_fim.pickle", "wb") as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Ended...")