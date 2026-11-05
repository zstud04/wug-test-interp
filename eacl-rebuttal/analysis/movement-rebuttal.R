# EACL rebuttal version of analysis/movement.R.
# Same plots, but the panels are the two kept image-only results
# (Qwen3-VL-2B and Qwen3-VL-4B, matched captions, 50 seeds each) instead of
# Language / Vision for a single model. CSVs come from
# eacl-rebuttal/scripts/movement_pca.py, copied into the git-tracked
# eacl-rebuttal/pca_data/ by eacl-rebuttal/scripts/export_pca_data.py.
#
# please open the .Rproj file in the home dir of
# the repo so that the paths resolve.

library(tidyverse)
library(scales)
library(ggnewscale)
library(ggdist)
library(ggtext)
library(grid)
library(glue)
library(knitr)

results_root = "eacl-rebuttal/pca_data"  # written by eacl-rebuttal/scripts/export_pca_data.py
models = c(
  "Qwen3-VL-2B" = "Qwen_Qwen3-VL-2B-Instruct",
  "Qwen3-VL-4B" = "Qwen_Qwen3-VL-4B-Instruct"
)

real_nouns <- imap_dfr(models, function(dir, model) {
  read_csv(glue("{results_root}/{dir}/sg_pl_reduced.csv")) %>%
    mutate(model = model)
}) %>%
  rename(number = label) %>%
  mutate(
    number = factor(number, c("sg", "pl")),
    model = factor(model, names(models))
  )

wug_pca <- imap_dfr(models, function(dir, model) {
  read_csv(glue("{results_root}/{dir}/wug_wugs_reduced_image.csv")) %>%
    mutate(model = model)
}) %>%
  mutate(model = factor(model, names(models)))

movement_pca_plot <- function(annot_size, strip_size = NULL) {
  wug_pca %>%
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
    geom_point(data = real_nouns, aes(x, y, color = number, shape = number), alpha = 0.08) +
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

    # --- 4. ANNOTATIONS (first panel only) ---
    geom_segment(
      data = data.frame(model = factor("Qwen3-VL-2B", names(models))),
      aes(x = 0.2, y = 0.2, xend = 0.28, yend = 0.3),
      arrow = arrow(length = unit(0.1, "cm")),
      color = "black", linewidth = 0.6
    ) +
    geom_text(
      data = data.frame(model = factor("Qwen3-VL-2B", names(models))),
      aes(x = 0.25, y = 0.18),
      label = "initial",
      family = "Times",
      fontface = "italic",
      size = annot_size
    ) +
    geom_text(
      data = data.frame(model = factor("Qwen3-VL-2B", names(models))),
      aes(x = 0.33, y = 0.3),
      label = "final",
      family = "Times",
      fontface = "italic",
      size = annot_size
    ) +
    scale_y_continuous(limits = c(-0.4, 0.4)) +
    scale_x_continuous(limits = c(-0.5, 0.5)) +
    # --- 5. THEME ---
    theme_classic(base_size = 16, base_family = "Times") +
    theme(
      panel.grid = element_blank(),
      strip.background = element_blank(),
      strip.text.x = element_text(face = 'bold.italic', size = strip_size),

      # Position legend at the top and stack the two legends horizontally
      legend.position = "top",
      legend.box = "horizontal",

      legend.text = element_text(size = 14, face = "italic"),
      legend.box.spacing = unit(0, "pt"),
      plot.margin = margin(0, 0, 0, 0, "pt"),

      # Transparency and no borders
      plot.background = element_rect(fill = "transparent", color = NA),
      panel.background = element_rect(fill = "transparent", color = NA),
      legend.background = element_rect(fill = "transparent", color = NA),
      legend.box.background = element_rect(fill = "transparent", color = NA),
      panel.border = element_blank()
    ) +
    labs(
      x = "PC1",
      y = "PC2"
    )
}

movement_pca_plot(annot_size = 3.5) +
  facet_wrap(~model, scales = "free")

ggsave("eacl-rebuttal/figures/2b-4b-movement-pca-legendtop.pdf", width = 6.8, height = 4.16, dpi = 300)

# ---

movement_pca_plot(annot_size = 4, strip_size = 14) +
  facet_wrap(~model, scales = "free", nrow = 2)

ggsave("eacl-rebuttal/figures/2b-4b-movement-pca-vertical.pdf", height = 7.95, width = 4.22, dpi = 300)

# ---

movement <- imap_dfr(models, function(dir, model) {
  read_csv(glue("{results_root}/{dir}/wug_wugs_movement_image.csv")) %>%
    mutate(model = model)
}) %>%
  mutate(model = factor(model, names(models)))

movement %>%
  group_by(model, number) %>%
  summarize(
    n = n(),
    sd = sd(movement, na.rm = TRUE),
    mean_movement = mean(movement, na.rm = TRUE),
    se = sd / sqrt(n),
    # 95% Confidence Interval
    ci_lower = mean_movement - qt(0.975, df = n - 1) * se,
    ci_upper = mean_movement + qt(0.975, df = n - 1) * se,

    # One-sample t-test (H0: mean = 0, HA: mean > 0)
    t_stat = mean_movement / se,
    p_value = pt(t_stat, df = n - 1, lower.tail = FALSE),

    .groups = "drop"
  )

movement %>%
  mutate(number = factor(number, levels = c("sg", "pl"))) %>%
  group_by(model, number) %>%
  summarize(
    n = n(),
    mean_movement = mean(movement, na.rm = TRUE),
    sd = sd(movement, na.rm = TRUE),
    me = qt(0.975, df = n - 1) * (sd / sqrt(n)),
    cell_value = sprintf("%.3f\\textsubscript{$\\pm$ %.3f}", mean_movement, me),
    .groups = "drop"
  ) %>%
  select(model, number, cell_value) %>%
  pivot_wider(
    names_from = number,
    values_from = cell_value
  ) %>%
  kable(
    format = "latex",
    booktabs = TRUE,
    escape = FALSE,
    caption = "Mean Movement (with 95\\% CI Margin of Error)"
  )

movement %>%
  mutate(number = factor(number, levels = c("sg", "pl"))) %>%
  ggplot(aes(number, movement, color = model, group = model, shape = model)) +
  geom_point(position = position_jitterdodge(dodge.width = 0.75, seed = 1024), alpha = 0.1) +
  stat_summary(
    fun.data = "mean_cl_normal",
    geom = "pointrange",
    position = position_dodge(width = 0.75),
    size = 0.4
  ) +
  geom_hline(yintercept = 0.0, linetype = "dashed") +
  # 0.25, not 0.2: one 2B seed moves 0.214, and scale limits drop points
  # before stat_summary computes the mean/CI.
  scale_y_continuous(limits = c(0, 0.25)) +
  theme_classic(base_size = 16, base_family = "Times") +
  labs(
    x = "Number",
    y = "Movement towards\ndesired region",
    color = "Model",
    shape = "Model"
  ) +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.95, 0.95),
    legend.justification = c("right", "top")
  )
