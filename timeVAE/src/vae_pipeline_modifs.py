import os, warnings
import numpy as np
import time


from data_utils import (
    load_yaml_file,
    load_data,
    split_data,
    scale_data,
    inverse_transform_data,
    save_scaler,
    save_data,
)
import paths
from vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_posterior_samples,
    get_prior_samples,
    load_vae_model,
)
from visualize import plot_samples, plot_latent_space_samples, visualize_and_save_tsne


# with or without the labels ?

def run_vae_pipeline(dataset_name: str, vae_type: str):

    # with open('/workspace/hyperparam-pine/timeVAE/src/outcomes.txt', 'r') as file:
    #     lines = file.readlines()
    # # Split each line into its components and convert to float
    # data = [list(map(float, line.strip().split())) for line in lines]
    # # Convert the list to a NumPy array
    # data_array = np.array(data)

    data_with = load_data(data_dir="/workspace/hyperparam-pine/", dataset="with_three_data")
    print(data_with.shape)

    # Separate the last element (11th element in dimension 2) to check its value

    # 25 au lieu de 10
    last_element = data_with[:, 25, 0]

    # print(last_element.shape)
    # print(last_element)

    # Create boolean masks for the condition
    mask_1 = last_element == 1
    mask_0 = last_element == 0

    # Select the first 10 elements of dimension 2 where the last element is 1
    #data_array_1 = np.expand_dims(data_array[mask_1][:, :25, :], axis=-1)
    data_with_1 = data_with[mask_1][:, :25, :]

    # Select the first 10 elements of dimension 2 where the last element is 0
    #data_array_0 = np.expand_dims(data_array[mask_0][:, :25, :], axis=-1)
    data_with_0 = data_with[mask_0][:, :25, :]



    # print("data_array_1")
    # print(data_array_1)
    # print("data_array_0")
    # print(data_array_0)

    # ----------------------------------------------------------------------------------
    # Load data, perform train/valid split, scale data

    # read data
    data = load_data(data_dir=paths.DATASETS_DIR, dataset=dataset_name)

    #print(data.shape) # (975, 25, 3)
    data = data[:, :25 , :]

    # split data into train/valid splits
    train_data, valid_data = split_data(data, valid_perc=0.05, shuffle=True)

    # scale data
    scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)

    data_with_0_normed = scaler.transform(data_with_0)
    data_with_1_normed = scaler.transform(data_with_1)


    # ----------------------------------------------------------------------------------
    # Instantiate and train the VAE Model

    # load hyperparameters from yaml file
    hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)[vae_type]

    # instantiate the model
    _, sequence_length, feature_dim = scaled_train_data.shape
    vae_model = instantiate_vae_model(
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        learning_rate=0.001,
        **hyperparameters,
    )

    # train vae
    train_vae(
        vae=vae_model,
        train_data=scaled_train_data,
        max_epochs=50,
        verbose=1,
    )

    # ----------------------------------------------------------------------------------
    # Save scaler and model
    model_save_dir = os.path.join(paths.MODELS_DIR, dataset_name)
    # save scaler
    save_scaler(scaler=scaler, dir_path=model_save_dir)
    # Save vae
    #save_vae_model(vae=vae_model, dir_path=model_save_dir)

    # ----------------------------------------------------------------------------------
    # Visualize posterior samples
    x_decoded = get_posterior_samples(vae_model, scaled_train_data)
    plot_samples(
        samples1=scaled_train_data,
        samples1_name="Original Train",
        samples2=x_decoded,
        samples2_name="Reconstructed Train",
        num_samples=5,
    )
    # ----------------------------------------------------------------------------------
    # Generate prior samples, visualize and save them

    # Generate prior samples
    prior_samples = get_prior_samples(vae_model, num_samples=train_data.shape[0])
    # Plot prior samples
    plot_samples(
        samples1=prior_samples,
        samples1_name="Prior Samples",
        num_samples=5,
    )

    ##### mouais non en fait

    #########   je pensais que ça allait faire une TSNE sur l'encoding bordel

    #########           PAS sur l'input

    #vae_model.encode()
    _, _, z_0 = vae_model.encoder(data_with_0_normed)
    _, _, z_1 = vae_model.encoder(data_with_1_normed)

    _, _, train_ = vae_model.encoder(scaled_train_data)
    _, _, set_ = vae_model.encoder(scaled_valid_data)

    print(z_0)
    print(z_0.shape) # (143, 10)
    print(scaled_train_data.shape)
    # (877, 25, 1)
    z_0 = np.expand_dims(z_0, axis=-1)
    z_1 = np.expand_dims(z_1, axis=-1)
    train_ = np.expand_dims(train_, axis=-1)
    set_ = np.expand_dims(set_, axis=-1)


    # visualize t-sne of original and prior samples
    visualize_and_save_tsne(
        samples1=train_, # scaled_train_data
        samples1_name="Original",
        samples2=set_, # prior_samples
        samples2_name="Generated (Prior)",
        samples3=z_0, # data_with_0_normed
        samples3_name="Sample3",
        samples4=z_1, # data_with_1_normed
        samples4_name="Sample4",
        scenario_name=f"Model-{vae_type} Dataset-{dataset_name}",
        save_dir=os.path.join(paths.TSNE_DIR, dataset_name),
        max_samples=2000,
    )

    exit()

    # inverse transformer samples to original scale and save to dir
    inverse_scaled_prior_samples = inverse_transform_data(prior_samples, scaler)
    save_data(
        data=inverse_scaled_prior_samples,
        output_file=os.path.join(
            os.path.join(paths.GEN_DATA_DIR, dataset_name),
            f"{vae_type}_{dataset_name}_prior_samples.npz",
        ),
    )

    # ----------------------------------------------------------------------------------
    # If latent_dim == 2, plot latent space
    if hyperparameters["latent_dim"] == 2:
        plot_latent_space_samples(vae=vae_model, n=8, figsize=(15, 15))

    # ----------------------------------------------------------------------------------
    # later.... load model
    loaded_model = load_vae_model(vae_type, model_save_dir)

    # Verify that loaded model produces same posterior samples
    new_x_decoded = loaded_model.predict(scaled_train_data)
    print(
        "Preds from orig and loaded models equal: ",
        np.allclose(x_decoded, new_x_decoded, atol=1e-5),
    )

    # ----------------------------------------------------------------------------------
# 

if __name__ == "__main__":
    # check `/data/` for available datasets
    #dataset = "sine_subsampled_train_perc_20"
    dataset = "all_three_data"
    # models: vae_dense, vae_conv, timeVAE
    model_name = "timeVAE"

    run_vae_pipeline(dataset, model_name)
