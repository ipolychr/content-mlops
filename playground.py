import ser.entrypoints as ml_entrypoints
from ser.data_model.criterion import Criterion, return_criterion
from ser.data_model.tl_models import TLModel
from ser.feature_extraction.factories import get_feature_extraction
from ser.preprocessor.factories import get_audio_preprocessor, get_dataset
from ser.trainer.optimizer import Optimizer

feature_extractor = get_feature_extraction()
img_data = get_dataset()
preprocessor = get_audio_preprocessor()

if __name__ == "__main__":
    # check create image features using audio files

    # ml_entrypoints.create_image_features_from_audio(path_with_audios_dir='Data/genres_original',
    #                                                 path_to_image='Test')

    # check train using original audios
    ml_entrypoints.train_tl_model_images(tl_model=TLModel.resnet34.value, criterion=return_criterion(Criterion.cross_entropy.name),
                                         optimizer=Optimizer.adam.value,
                                         checkpoints_path="ser_checkpoints", num_epoch=25, images_path='to_run',
                                         )
