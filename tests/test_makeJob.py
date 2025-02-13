# from tests.conftest import *
# import pathlib
# import pytest
# from typing import Sequence

# def test_makeStateJob_returns_path_when_writeJob_is_true(temporaryDirectoryPath: pathlib.Path, list_dimensions: Sequence[int] = (11, 13)) -> None:
# 	"""
# 	Test that makeStateJob returns a Path object when writeJob is True.
# 	"""
# 	pathFilenameJob = makeStateJob(list_dimensions)
# 	assert isinstance(pathFilenameJob, pathlib.Path), "Expected a Path object to be returned."
# 	assert pathFilenameJob.exists(), "Expected the file to exist."

# def test_makeStateJob_returns_computationState_when_writeJob_is_false(list_dimensions: Sequence[int] = (11, 13)) -> None:
# 	"""
# 	Test that makeStateJob returns a computationState object when writeJob is False.
# 	"""
# 	stateUniversal = makeStateJob(list_dimensions, writeJob=False)
# 	assert isinstance(stateUniversal, computationState), "Expected a computationState object to be returned."

# def test_makeStateJob_writes_file_to_correct_location(temporaryDirectoryPath: pathlib.Path, list_dimensions: Sequence[int] = (11, 13)) -> None:
# 	"""
# 	Test that makeStateJob writes the job file to the correct location.
# 	"""
# 	pathFilenameJob = makeStateJob(list_dimensions)
# 	expectedPath = getPathFilenameFoldsTotal(list_dimensions)
# 	suffix = expectedPath.suffix
# 	expectedPath = pathlib.Path(str(expectedPath)[0:-len(suffix)]) / 'stateJob.pkl'
# 	assert pathFilenameJob.resolve() == expectedPath.resolve(), "The file was written to an unexpected location."

# def test_makeStateJob_correctly_passes_keyword_arguments(temporaryDirectoryPath: pathlib.Path) -> None:
# 	"""
# 	Test that makeStateJob correctly passes keyword arguments to outfitCountFolds.
# 	"""
# 	listDimensions = [11, 13]
# 	pathFilenameJob = makeStateJob(listDimensions, algorithmName="cardinal")
# 	pathFilenameJob = makeStateJob(listDimensions, writeJob=False, algorithmName="cardinal")
# 	#Cannot directly verify that outfitCountFolds receives the keyword argument.
# 	#This test confirms that the makeStateJob function does not crash when passing the keyword argument.
# 	assert True # If the test reaches here without an error, it is considered a pass.

# def test_makeStateJob_raises_error_with_invalid_dimension(temporaryDirectoryPath: pathlib.Path) -> None:
# 	"""
# 	Test that makeStateJob raises a ValueError when an invalid dimension is passed.
# 	"""
# 	with pytest.raises(ValueError):
# 		makeStateJob([7, "cardinal"])

# def test_makeStateJob_creates_directory_structure(temporaryDirectoryPath: pathlib.Path, list_dimensions: Sequence[int] = (11, 13)) -> None:
# 	"""
# 	Test that makeStateJob creates the directory structure for the job.
# 	"""
# 	pathFilenameJob = makeStateJob(list_dimensions)
# 	directoryPath = pathFilenameJob.parent
# 	assert directoryPath.exists(), "The directory structure was not created."

# def test_makeStateJob_writes_valid_pickle_file(temporaryDirectoryPath: pathlib.Path, list_dimensions: Sequence[int] = (11, 13)) -> None:
# 	"""
# 	Test that makeStateJob writes a valid pickle file.
# 	"""
# 	pathFilenameJob = makeStateJob(list_dimensions)
# 	with open(pathFilenameJob, 'rb') as readStreamBinaryMode:
# 		try:
# 			pickle.load(readStreamBinaryMode)
# 		except Exception as ERRORmessage:
# 			pytest.fail(f"The pickle file is invalid: {ERRORmessage}")
