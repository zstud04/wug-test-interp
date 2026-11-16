# EACL rebuttal version of analysis/movement.R (4B) and analysis/movement-2B.R (2B):
# same figures, sizes and file names, with the Language panel unchanged (the
# paper's language-cued embeddings, results/eval/movement-analysis/) and the
# Vision panel replaced by the NEW vision-cued embeddings (matched captions,
# held-out-image dev, 50 seeds; eacl-rebuttal/pca_data/, written by
# eacl-rebuttal/scripts/movement_pca.py + export_pca_data.py). Both use the
# same PCA space (same 530 noun pairs and model; identical up to bf16 rounding).
#
# please open the .Rproj file in the home dir of
# the repo so that the paths resolve.
# On the server: /opt/R/4.6.2/bin/Rscript eacl-rebuttal/analysis/movement-rebuttal.R

library(tidyverse)
library(scales)
library(ggnewscale)
library(ggdist)
library(ggtext)
library(grid)
library(glue)
library(knitr)

paper_root = "results/eval/movement-analysis"
new_root = "eacl-rebuttal/pca_data"
fig_dir = "eacl-rebuttal/figures"

load_model <- function(dir) {
  real_nouns <- read_csv(glue("{paper_root}/{dir}/sg_pl_reduced.csv"), show_col_types = FALSE) %>%
    rename(number = label) %>%
    mutate(
      number = factor(number, c("sg", "pl"))
    )

  wug_pca <- bind_rows (
    read_csv(glue("{paper_root}/{dir}/wug_wugs_reduced_text.csv"), show_col_types = FALSE) %>%
      mutate(modality='Language'),
    read_csv(glue("{new_root}/{dir}/wug_wugs_reduced_image.csv"), show_col_types = FALSE) %>%
      mutate(modality='Vision')
  )

  movement <- bind_rows (
    read_csv(glue("{paper_root}/{dir}/wug_wugs_movement_text.csv"), show_col_types = FALSE) %>%
      mutate(modality='Language'),
    read_csv(glue("{new_root}/{dir}/wug_wugs_movement_image.csv"), show_col_types = FALSE) %>%
      mutate(modality='Vision')
  )

  list(real_nouns = real_nouns, wug_pca = wug_pca, movement = movement)
}

# Plot code from analysis/movement.R / movement-2B.R. `fixed_limits` is TRUE for
# 2B (scale limits set in movement-2B.R) and FALSE for 4B (commented out in movement.R).
movement_pca_plot <- function(d, annot_size, nrow = NULL, strip_size = NULL, fixed_limits) {
  p <- d$wug_pca %>%
    pivot_wider(names_from = stage, values_from = c(x, y)) %>%
    mutate(
      type = case_when(
        type == "wug" ~ "sg",
        TRUE ~ "pl"
      ),
      type = factor(type, c("sg", "pl"))
    ) %>%
    ggplot() +

    # --- 1. REAL NOUNS ---
    geom_point(data = d$real_nouns, aes(x, y, color = number, shape = number), alpha = 0.08) +
    scale_color_manual(
      name = "Real Noun Type",
      values = c("#a6611a", "#018571"),
      # Define guide directly inside the scale to avoid ggnewscale errors
      guide = guide_legend(
        title.position = "top",
        title.hjust = 0.5,
        override.aes = list(alpha = 1)
      )
    ) +
    scale_shape_manual(
      name = "Real Noun Type",
      values = c(16, 17),
      guide = guide_legend(
        title.position = "top",
        title.hjust = 0.5,
        override.aes = list(alpha = 1)
      )
    ) +

    # --- 2. RESET SCALES ---
    new_scale_color() +

    # --- 3. NOVEL NOUNS ---
    geom_segment(
      aes(
        x = x_init, y = y_init,
        xend = x_final, yend = y_final,
        color = type
      ),
      arrow = arrow(length = unit(0.1, "cm")),
      linewidth = 0.6, alpha = 0.6
    ) +
    scale_color_manual(
      name = "Novel Noun Type",
      values = c("#a6611a", "#018571"),
      guide = guide_legend(
        title.position = "top",
        title.hjust = 0.5,
        override.aes = list(alpha = 1, linewidth = 0.6)
      )
    ) +

    # --- 4. ANNOTATIONS ---
    geom_segment(
      data = data.frame(modality = "Language"),
      aes(x = 0.2, y = 0.2, xend = 0.28, yend = 0.3),
      arrow = arrow(length = unit(0.1, "cm")),
      color = "black", linewidth = 0.6
    ) +
    geom_text(
      data = data.frame(modality="Language"),
      aes(x = 0.25, y = 0.18),
      label = "initial",
      family = "Times",
      fontface = "italic",
      size = annot_size
    ) +
    geom_text(
      data = data.frame(modality="Language"),
      aes(x = 0.33, y = 0.3),
      label = "final",
      family = "Times",
      fontface = "italic",
      size = annot_size
    )

  if (fixed_limits) {
    p <- p +
      scale_y_continuous(limits = c(-0.4, 0.4)) +
      scale_x_continuous(limits = c(-0.5, 0.5))
  }

  p +
    # --- 5. THEME & FACETS ---
    facet_wrap(~modality, scales="free", nrow = nrow) +
    theme_classic(base_size = 16, base_family = "Times") +
    theme(
      panel.grid = element_blank(),
      strip.background = element_blank(),
      strip.text.x = element_text(face='bold.italic', size = strip_size),

      # Position legend at the top and stack the two legends horizontally
      legend.position = "top",
      legend.box = "horizontal",

      # Target legend labels with Inconsolata
      legend.text = element_text(size = 14, face = "italic"),
      legend.box.spacing = unit(0, "pt"),
      plot.margin = margin(0, 0, 0, 0, "pt"),

      # --- New code for transparency and no borders ---
      plot.background = element_rect(fill = "transparent", color = NA), # Transparent canvas, no border
      panel.background = element_rect(fill = "transparent", color = NA), # Transparent plot area
      legend.background = element_rect(fill = "transparent", color = NA), # Transparent legend
      legend.box.background = element_rect(fill = "transparent", color = NA),
      panel.border = element_blank() # Removes any leftover panel borders
    ) +
    labs(
      x = "PC1",
      y = "PC2"
    )
}

models <- list(
  "2b" = list(dir = "Qwen_Qwen3-VL-2B-Instruct", fixed_limits = TRUE),
  "4b" = list(dir = "Qwen_Qwen3-VL-4B-Instruct", fixed_limits = FALSE)
)

summaries <- list()
for (short in names(models)) {
  cfg <- models[[short]]
  d <- load_model(cfg$dir)

  movement_pca_plot(d, annot_size = 3.5, fixed_limits = cfg$fixed_limits)
  ggsave(glue("{fig_dir}/{short}-movement-pca-legendtop.pdf"), width = 6.8, height = 4.16, dpi = 300)

  movement_pca_plot(d, annot_size = 4, nrow = 2, strip_size = 14, fixed_limits = cfg$fixed_limits)
  ggsave(glue("{fig_dir}/{short}-movement-pca-vertical.pdf"), height = 7.95, width = 4.22, dpi = 300)

  # Movement summary, as in movement.R
  summaries[[short]] <- d$movement %>%
    mutate(number = factor(number, levels = c("sg", "pl"))) %>%
    group_by(modality, number) %>%
    summarize(
      n = n(),
      sd = sd(movement, na.rm = TRUE),
      mean_movement = mean(movement, na.rm = TRUE),
      # Calculate Standard Error
      se = sd / sqrt(n),
      # 95% Confidence Interval
      ci_lower = mean_movement - qt(0.975, df = n - 1) * se,
      ci_upper = mean_movement + qt(0.975, df = n - 1) * se,
      # One-sample t-test (H0: mean = 0, HA: mean > 0)
      t_stat = mean_movement / se,
      p_value = pt(t_stat, df = n - 1, lower.tail = FALSE),
      .groups = "drop"
    ) %>%
    mutate(model = toupper(short), .before = 1)
}

summary_tab <- bind_rows(summaries)
write_csv(summary_tab, glue("{new_root}/movement_summary_language_vision.csv"))
print(summary_tab, width = Inf)

# LaTeX table in the paper's format (mean with 95% CI margin), per model
for (short in names(models)) {
  print(
    summary_tab %>%
      filter(model == toupper(short)) %>%
      mutate(
        me = ci_upper - mean_movement,
        cell_value = sprintf("%.3f\\textsubscript{$\\pm$ %.3f}", mean_movement, me)
      ) %>%
      select(modality, number, cell_value) %>%
      pivot_wider(names_from = number, values_from = cell_value) %>%
      kable(
        format = "latex",
        booktabs = TRUE,
        escape = FALSE,
        caption = glue("Mean Movement (with 95\\% CI Margin of Error), Qwen3-VL-{toupper(short)}")
      )
  )
}
