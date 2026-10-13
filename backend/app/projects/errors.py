"""Project and source-image errors.

Messages returned to clients are chosen in the route layer so filesystem
paths never leak into the response.
"""


class ProjectError(Exception):
    pass


class ProjectNotFound(ProjectError):
    pass


class SourceImageNotFound(ProjectError):
    pass


class ProjectConflict(ProjectError):
    pass


class ProjectDuplicateTooLarge(ProjectError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class ProjectDuplicateFailed(ProjectError):
    def __init__(self, detail: str = "The project could not be duplicated.") -> None:
        self.detail = detail
        super().__init__(detail)


class InvalidProjectName(ProjectError):
    pass


class UnsafeProjectPath(ProjectError):
    pass


class InvalidSourceImage(ProjectError):
    pass


class UnsupportedSourceImage(ProjectError):
    pass


class SourceImageTooLarge(ProjectError):
    pass


class DepthModelUnavailable(ProjectError):
    pass


class DepthInferenceFailed(ProjectError):
    pass


class DepthArtifactNotFound(ProjectError):
    pass


class RawDepthNotFound(ProjectError):
    pass


class ConditioningFailed(ProjectError):
    pass


class ConditionedPreviewNotFound(ProjectError):
    pass


class ConditionedDepthNotFound(ProjectError):
    pass


class SegmentationModelUnavailable(ProjectError):
    pass


class SegmentationFailed(ProjectError):
    pass


class InvalidSegmentationPrompt(ProjectError):
    pass


class SegmentationProposalNotFound(ProjectError):
    pass


class RefinementSessionNotFound(ProjectError):
    pass


class InvalidMaskStroke(ProjectError):
    pass


class MaskRefinementFailed(ProjectError):
    pass


class InvalidMagicWand(ProjectError):
    pass


class MagicWandFailed(ProjectError):
    pass


class ReliefRegionNotFound(ProjectError):
    pass


class ReliefCompositionFailed(ProjectError):
    pass


class MaskSaveFailed(ProjectError):
    pass


class InvalidReliefRequest(ProjectError):
    pass


class ComposedReliefNotFound(ProjectError):
    pass


class InvalidPhysicalHeight(ProjectError):
    def __init__(self, reason: str = "parameters") -> None:
        self.reason = reason
        super().__init__(reason)


class PhysicalPreviewNotFound(ProjectError):
    pass


class PhysicalHeightNotFound(ProjectError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class InvalidPhysicalHeightArtifact(ProjectError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class InvalidProfileRequest(ProjectError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class InvalidMeshRequest(ProjectError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class InvalidExportRequest(ProjectError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class PhysicalHeightFailed(ProjectError):
    pass


class ArtifactDependencyRejected(ProjectError):
    """A processing step needs a verified current upstream artifact."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)
