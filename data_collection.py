import pickle
from tournament.runner import main as runner_main

data_files = []
agents = ["AI", "state_agent", "image_agent", "geoffrey_agent", "image_jurgen_agent", "jurgen_agent", "yann_agent", "yoshua_agent"]

for agent in agents:
    for i in range(10): 
        output_file = f"{agent}_data_{i}.pkl"
        data_files.append(output_file)
        runner_args = ["image_agent", agent, "-s", output_file]
        runner_main(runner_args)

combined_data = []

for data_file in data_files:
    with open(data_file, "rb") as f:
        data = pickle.load(f)
        combined_data.extend(data)

with open("combined_data.pkl", "wb") as f:
    pickle.dump(combined_data, f)
