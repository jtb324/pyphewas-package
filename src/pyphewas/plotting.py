import numpy as np
from pyphewas.parser import generate_parser
import polars as pl
from matplotlib import colormaps
import adjustText
import seaborn as sns


def generate_plot() -> None:
    data = pl.read_csv(
        "/data100t1/home/james/code_projects/PyPheWAS/tests/output/potential_mito_cases_SD_phewas_7_3_25_v4.txt.gz",
        separator="\t",
        columns=[
            "phecode",
            "phecode_description",
            "phecode_category",
            "case_count",
            "control_count",
            "converged",
            "status_pvalue",
        ],
    )

    print(data.head())


class PlotBuilder: ...


def _create_phecode_index(df) -> pl.DataFrame:
    """
    Create phecode index after grouping by phecode_category and phecode;
    Phecode index will be used for plotting purpose
    :param df: PheWAS result to create index
    :return: same dataframe with column "phecode_index" created
    """
    if "phecode_index" in df.columns:
        df = df.drop("phecode_index")

    df = (
        df.sort(by=["phecode_category", "phecode"])
        .with_columns(pl.Series("phecode_index", range(1, len(df) + 1)))
        .with_columns(15 * np.exp(pl.col("beta")).alias("marker_size_by_beta"))
    )

    return df


def _x_ticks(plot_df, selected_color_dict, size=8) -> None:
    """
    Generate x tick labels and colors
    :param plot_df: plot data
    :param selected_color_dict: color dict; this is changed based on number of phecode categories selected
    :return: x tick labels and colors for the plot
    """
    x_ticks = (
        plot_df[["phecode_category", "phecode_index"]]
        .group_by("phecode_category")
        .mean()
    )
    # create x ticks labels and colors
    adjustText.plt.xticks(
        x_ticks["phecode_index"],
        x_ticks["phecode_category"],
        rotation=45,
        ha="right",
        weight="normal",
        size=size,
    )

    tick_labels = adjustText.plt.gca().get_xticklabels()
    sorted_labels = sorted(tick_labels, key=lambda label: label.get_text())
    for tick_label, tick_color in zip(sorted_labels, selected_color_dict.values()):
        tick_label.set_color(tick_color)


def _manhattan_scatter(self, ax, marker_size_by_beta, scale_factor=1):
    """
    Generate scatter data points
    :param ax: plot object
    :param marker_size_by_beta: adjust marker size by beta coefficient if True
    :return: scatter plot of selected data
    """

    if marker_size_by_beta:
        s_positive = self.positive_betas["_marker_size"] * scale_factor
        s_negative = self.negative_betas["_marker_size"] * scale_factor
    else:
        s_positive = None
        s_negative = None

    ax.scatter(
        x=self.positive_betas["phecode_index"].to_numpy(),
        y=self.positive_betas["neg_log_p_value"],
        s=s_positive,
        c=self.positive_betas["label_color"],
        marker="^",
        alpha=self.positive_alpha,
    )

    ax.scatter(
        x=self.negative_betas["phecode_index"].to_numpy(),
        y=self.negative_betas["neg_log_p_value"],
        s=s_negative,
        c=self.negative_betas["label_color"],
        marker="v",
        alpha=self.negative_alpha,
    )


def _lines(
    self,
    ax,
    plot_type,
    plot_df,
    x_col,
    nominal_significance_line=False,
    bonferroni_line=False,
    infinity_line=False,
    y_threshold_line=False,
    y_threshold_value=None,
    x_positive_threshold_line=False,
    x_positive_threshold_value=None,
    x_negative_threshold_line=False,
    x_negative_threshold_value=None,
):

    extra_offset = 0
    if plot_type == "manhattan":
        extra_offset = 1
    elif plot_type == "volcano":
        extra_offset = 0.05

    # nominal significance line
    if nominal_significance_line:
        ax.hlines(
            y=-adjustText.np.log10(0.05),
            xmin=plot_df[x_col].min() - self.offset - extra_offset,
            xmax=plot_df[x_col].max() + self.offset + extra_offset,
            colors="red",
            lw=1,
        )

    # bonferroni
    if bonferroni_line:
        ax.hlines(
            y=self.bonferroni,
            xmin=plot_df[x_col].min() - self.offset - extra_offset,
            xmax=plot_df[x_col].max() + self.offset + extra_offset,
            colors="green",
            lw=1,
        )

    # infinity
    if infinity_line:
        if self.inf_proxy is not None:
            ax.yaxis.get_major_ticks()[-2].set_visible(False)
            ax.hlines(
                y=self.inf_proxy * 0.98,
                xmin=plot_df[x_col].min() - self.offset - extra_offset,
                xmax=plot_df[x_col].max() + self.offset + extra_offset,
                colors="blue",
                linestyle="dashdot",
                lw=1,
            )

    # y threshold line
    if y_threshold_line:
        ax.hlines(
            y=y_threshold_value,
            xmin=plot_df[x_col].min() - self.offset - extra_offset,
            xmax=plot_df[x_col].max() + self.offset + extra_offset,
            colors="gray",
            linestyles="dashed",
            lw=1,
        )

    # vertical lines
    if x_positive_threshold_line:
        ax.vlines(
            x=x_positive_threshold_value,
            ymin=plot_df["neg_log_p_value"].min() - self.offset,
            ymax=plot_df["neg_log_p_value"].max() + self.offset + 5,
            colors="orange",
            linestyles="dashed",
            lw=1,
        )
    if x_negative_threshold_line:
        ax.vlines(
            x=x_negative_threshold_value,
            ymin=plot_df["neg_log_p_value"].min() - self.offset,
            ymax=plot_df["neg_log_p_value"].max() + self.offset + 5,
            colors="lightseagreen",
            linestyles="dashed",
            lw=1,
        )


def _split_text(s, threshold=30):
    """
    Split long text label
    :param s: text string
    :param threshold: approximate number of characters per line
    :return: split text if longer than 40 chars
    """
    words = s.split(" ")
    new_s = ""
    line_length = 0
    for word in words:
        new_s += word
        line_length += len(word)
        if line_length >= threshold and word != words[-1]:
            new_s += "\n"
            line_length = 0
        else:
            new_s += " "

    return new_s


def _manhattan_label(
    self,
    plot_df,
    label_values,
    label_count,
    label_categories=None,
    label_text_column="phecode_string",
    label_value_threshold=0,
    label_split_threshold=30,
    label_color="label_color",
    label_size=8,
    label_weight="normal",
    y_col="neg_log_p_value",
    x_col="phecode_index",
):
    """
    :param plot_df: plot data
    :param label_values: can take a single phecode, a list of phecodes,
                            or preset values "positive_betas", "negative_betas", "p_value"
    :param label_value_threshold: cutoff value for label values;
                                    if label_values is "positive_beta", keep beta values >= cutoff
                                    if label_values is "negative_beta", keep beta values <= cutoff
                                    if label_values is "p_value", keep neg_log_p_value >= cutoff
    :param label_text_column: defaults to "phecode_string"; name of column contain text for labels
    :param label_count: number of items to label, only needed if label_by input is data type
    :param label_split_threshold: number of characters to consider splitting long labels
    :param label_color: string type; takes either a color or name of column contains color for plot data
    :param label_size: defaults to 8
    :param label_weight: takes standard plt weight inputs, e.g., "normal", "bold", etc.
    :param x_col: column contains x values
    :param y_col: column contains y values
    :return: adjustText object
    """

    if isinstance(label_values, str):
        label_values = [label_values]

    self.data_to_label = pl.DataFrame(schema=plot_df.schema)
    positive_betas = self.positive_betas.clone()
    negative_betas = self.negative_betas.clone()
    if "_marker_size" in positive_betas.columns:
        positive_betas = positive_betas.drop("_marker_size")
    if "_marker_size" in negative_betas.columns:
        negative_betas = negative_betas.drop("_marker_size")

    for item in label_values:
        if item == "positive_beta":
            self.data_to_label = pl.concat(
                [
                    self.data_to_label,
                    positive_betas.filter(pl.col("beta") >= label_value_threshold),
                ]
            )
            if label_categories is not None:
                self.data_to_label = self.data_to_label.filter(
                    pl.col("phecode_category").is_in(label_categories)
                )[:label_count]
            else:
                self.data_to_label = self.data_to_label[:label_count]
        elif item == "negative_beta":
            self.data_to_label = pl.concat(
                [
                    self.data_to_label,
                    negative_betas.filter(pl.col("beta") <= label_value_threshold),
                ]
            )
            if label_categories is not None:
                self.data_to_label = self.data_to_label.filter(
                    pl.col("phecode_category").is_in(label_categories)
                )[:label_count]
            else:
                self.data_to_label = self.data_to_label[:label_count]
        elif item == "p_value":
            self.data_to_label = pl.concat(
                [
                    self.data_to_label,
                    plot_df.sort(by="p_value").filter(
                        pl.col("neg_log_p_value") >= label_value_threshold
                    ),
                ]
            )
            if label_categories is not None:
                self.data_to_label = self.data_to_label.filter(
                    pl.col("phecode_category").is_in(label_categories)
                )[:label_count]
            else:
                self.data_to_label = self.data_to_label[:label_count]
        else:
            self.data_to_label = pl.concat(
                [self.data_to_label, plot_df.filter(pl.col("phecode") == item)]
            )

    texts = []
    for i in range(len(self.data_to_label)):
        if mc.is_color_like(label_color):
            color = pl.Series(values=[label_color] * len(self.data_to_label))
        else:
            # noinspection PyTypeChecker
            color = self.data_to_label[label_color]
        # noinspection PyTypeChecker
        texts.append(
            adjustText.plt.text(
                float(self.data_to_label[x_col][i]),
                float(self.data_to_label[y_col][i]),
                self._split_text(
                    self.data_to_label[label_text_column][i], label_split_threshold
                ),
                color=color[i],
                size=label_size,
                weight=label_weight,
                alpha=1,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    boxstyle="round",
                    alpha=0.5,
                    lw=0.5,
                ),
            )
        )

    if len(texts) > 0:
        return adjustText.adjust_text(
            texts,
            arrowprops=dict(
                arrowstyle="simple", color="gray", lw=0.5, mutation_scale=2
            ),
        )


def _manhattan_legend(self, ax, legend_marker_size):
    """
    :param ax: plot object
    :param legend_marker_size: size of markers
    :return: legend element
    """
    legend_elements = [
        Line2D([0], [0], color="blue", lw=1, linestyle="dashdot", label="Infinity"),
        Line2D([0], [0], color="green", lw=1, label="Bonferroni\nCorrection"),
        Line2D([0], [0], color="red", lw=1, label="Nominal\nSignificance"),
        Line2D(
            [0],
            [0],
            marker="^",
            label="Increased\nRisk Effect",
            color="white",
            markerfacecolor="blue",
            alpha=self.positive_alpha,
            markersize=legend_marker_size,
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            label="Decreased\nRisk Effect",
            color="white",
            markerfacecolor="blue",
            alpha=self.negative_alpha,
            markersize=legend_marker_size,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        handlelength=2,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=legend_marker_size,
    )


def manhattan(
    self,
    label_values="p_value",
    label_value_threshold=0,
    label_count=10,
    label_size=8,
    label_text_column="phecode_string",
    label_color="label_color",
    label_weight="normal",
    label_split_threshold=30,
    marker_size_by_beta=False,
    marker_scale_factor=1,
    phecode_categories=None,
    plot_all_categories=True,
    title=None,
    title_text_size=10,
    y_limit=None,
    axis_text_size=8,
    show_legend=True,
    legend_marker_size=6,
    dpi=150,
    save_plot=True,
    output_file_name=None,
    output_file_type="pdf",
):

    ############
    # SETTINGS #
    ############

    # setup some variables based on plot_all_categories and phecode_categories

    # offset
    self.offset = 9

    # phecode_categories & label_categories
    if phecode_categories:
        if isinstance(phecode_categories, str):  # convert to list if input is str
            phecode_categories = [phecode_categories]
        phecode_categories.sort()
        label_categories = phecode_categories
        self.phecode_categories = phecode_categories
    else:
        label_categories = None

    # plot_df and label_value_cols
    if plot_all_categories:
        selected_color_dict = self.color_dict
        n_categories = len(self.phewas_result.columns)
        # create plot_df containing only necessary data for plotting
        plot_df = self._create_phecode_index(self.phewas_result)
    else:
        if phecode_categories:
            selected_color_dict = {k: self.color_dict[k] for k in phecode_categories}
            n_categories = len(phecode_categories)
            dpi = None
            # create plot_df containing only necessary data for plotting
            plot_df = self._create_phecode_index(
                self._filter_by_phecode_categories(
                    self.phewas_result, phecode_categories=phecode_categories
                )
            )
        else:
            print(
                "phecode_categories must not be None when plot_all_categories = False."
            )
            return

    # create plot
    self.ratio = n_categories / len(self.phewas_result.columns)
    fig, ax = adjustText.plt.subplots(figsize=(12 * self.ratio, 7), dpi=dpi)

    # plot title
    if title is not None:
        adjustText.plt.title(title, weight="bold", size=title_text_size)

    # set limit for display on y axes
    if y_limit is not None:
        ax.set_ylim(-0.2, y_limit)

    # y axis label
    ax.set_ylabel(r"$-\log_{10}$(p-value)", size=axis_text_size)

    # generate positive & negative betas
    self.positive_betas, self.negative_betas = self._split_by_beta(
        plot_df, marker_size_by_beta
    )

    ############
    # PLOTTING #
    ############

    # x-axis offset
    adjustText.plt.xlim(
        float(plot_df["phecode_index"].min()) - self.offset - 1,
        float(plot_df["phecode_index"].max()) + self.offset + 1,
    )

    # create x ticks labels and colors
    self._x_ticks(plot_df, selected_color_dict)

    # scatter
    self._manhattan_scatter(
        ax=ax, marker_size_by_beta=marker_size_by_beta, scale_factor=marker_scale_factor
    )

    # lines
    self._lines(
        ax=ax,
        plot_type="manhattan",
        plot_df=plot_df,
        x_col="phecode_index",
        nominal_significance_line=True,
        bonferroni_line=True,
        infinity_line=True,
    )

    # labeling
    self._manhattan_label(
        plot_df=plot_df,
        label_values=label_values,
        label_count=label_count,
        label_text_column=label_text_column,
        label_categories=label_categories,
        label_value_threshold=label_value_threshold,
        label_split_threshold=label_split_threshold,
        label_size=label_size,
        label_color=label_color,
        label_weight=label_weight,
    )

    # legend
    if show_legend:
        self._manhattan_legend(ax, legend_marker_size)

    # save plot
    if save_plot:
        self.save_plot(
            plot_type="manhattan",
            output_file_name=output_file_name,
            output_file_type=output_file_type,
        )


def generate_plot() -> None:
    # bonferroni
    # parser = generate_parser()

    # args = parser.parse_args()

    phewas_result = pl.read_csv(
        args.input,
        separator="\t",
        columns=[
            "phecode",
            "phecode_description",
            "phecode_category",
            "coverged",
            "status_pvalue",
            "status_beta",
        ],
    )

    # filter to converged results
    # if args.converged_only:
    phewas_result = phewas_result.filter(pl.col("converged") == "true")
    # lets also generate a -log10 pvalue column
    phewas_results = phewas_result.with_columns(
        -np.log10(pl.col("status_pvalue")).alias("negative_log10_pvalue")
    )

    phewas_results = _create_phecode_index(phewas_result)

    sns.relplot(
        data=phewas_results,
        x="phecode_index",
        y="negative_log10_pvalue",
        aspect=4,
        hue="phecode_category",
        palette="Set1",
    )

    # if args.bonferroni is None:
    #     bonferroni = -np.log10(0.05 / len(phewas_result))
    # else:
    #     bonferroni = args.bonferroni

    # nominal_significance = -np.log10(0.05)

    # cmap = colormaps["tab20"]

    # phecode_categories = sorted(phewas_result["phecode_category"].unique().to_list())

    # color_dict = {category: cmap(indx) for indx, category in enumerate(phecode_categories)}


if __name__ == "__main__":
    generate_plot()
