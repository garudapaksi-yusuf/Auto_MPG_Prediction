from configs import (
    BODY_SIDEBAR,
    BODY_TITLE,
    CSS_CUSTOM,
    FEATURE_SPECS,
    FIRST_INDEX,
    LAST_INDEX,
    MODEL_COLUMNS,
    MODEL_PATH,
    SPEC_COLUMNS,
    TEXT_ALIGNMENT_CUSTOM,
    WIDTH_CUSTOM,
    X_TRAIN,
    FeatureSpec,
)
from ui_helpers import (
    number_input_widget,
    page_layouts,
    slider_widget,
    widget_column_container,
    widget_session_state,
)


def render_feature_widget(spec: FeatureSpec, widget_column) -> None:
    min_value = X_TRAIN[spec.column].min()
    max_value = X_TRAIN[spec.column].max()
    common_kwargs = {
        "description": spec.label,
        "min_value": min_value,
        "max_value": max_value,
        "key": spec.column,
    }

    with widget_column: # context manager for widget_column
        if spec.widget_type == "number":
            value = number_input_widget(
                step=spec.step,
                format_custom=spec.format_custom,
                **common_kwargs,
            )
        else:
            value = slider_widget(
                step=spec.step,
                format_custom=spec.format_custom,
                integer=spec.integer,
                **common_kwargs,
            )

    widget_session_state(spec.column, value)


def render_feature_layout(widget_columns) -> list[str]:
    for index, spec in enumerate(FEATURE_SPECS):
        render_feature_widget(spec, widget_columns[index % len(widget_columns)])
    return MODEL_COLUMNS


def main() -> None:
    _, _, _, widget_columns = page_layouts(
        CSS_CUSTOM,
        BODY_SIDEBAR,
        WIDTH_CUSTOM,
        BODY_TITLE,
        TEXT_ALIGNMENT_CUSTOM,
        SPEC_COLUMNS,
    )

    columns = render_feature_layout(widget_columns)

    with widget_columns[LAST_INDEX]:
        widget_column_container(columns, WIDTH_CUSTOM, str(MODEL_PATH), FIRST_INDEX)


if __name__ == "__main__":
    main()