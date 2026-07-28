# please open the .Rproj file in the home dir of 
# the repo so that the paths resolve.

library(tidyverse)
library(scales)
library(ggnewscale)
library(ggdist)
library(ggtext)
library(grid)
library(glue)

# results_dir = "results/movement-analysis/Qwen_Qwen3-VL-2B-Instruct"
results_dir = "results/eval/movement-analysis/Qwen_Qwen3-VL-4B-Instruct"

real_nouns <- read_csv(glue("{results_dir}/sg_pl_reduced.csv")) %>%
  rename(number = label)

real_nouns %>%
  ggplot(aes(x,y, color = number)) +
  geom_point(alpha = 0.1) +
  scale_color_manual(name = "Known Noun\nType", values = c("#018571", "#a6611a")) +
  theme_classic(base_size = 17)+
  theme(
    axis.ticks = element_blank(),
    axis.title = element_blank(),
    legend.position = "none",
    axis.text = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background  = element_rect(fill = "transparent", colour = NA)
  )

ggsave("figures/demo-sg-pl-pca.svg", height = 2.15, width = 2.39, dpi = 300)


wug_pca <- bind_rows (
  read_csv(glue("{results_dir}/wug_wugs_reduced_text.csv")) %>%
    mutate(modality='Language'),
  read_csv(glue("{results_dir}/wug_wugs_reduced_image.csv")) %>%
    mutate(modality='Vision')
)

# font_add_google("Inconsolata", "Inconsolata")
# showtext_auto()


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
    values = c("#018571", "#a6611a"),
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
    values = rev(c("#018571", "#a6611a")),
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
    size = 4
  ) +
  geom_text(
    data = data.frame(modality="Language"),
    aes(x = 0.33, y = 0.3),
    label = "final",
    family = "Times", 
    fontface = "italic",
    size = 4
  ) +
  
  # --- 5. THEME & FACETS ---
  facet_wrap(~modality, scales="free") + 
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_text(face='bold.italic'),
    
    # Position legend at the top and stack the two legends horizontally
    legend.position = "top",
    legend.box = "horizontal",
    
    # Target legend labels with Inconsolata
    legend.text = element_text(family = "Inconsolata", size = 14) 
  ) +
  labs(
    x = "PC1",
    y = "PC2"
  )

# ggsave("figures/4b-movement-pca.pdf", width = 6.81, height = 3.9, dpi = 300, device = cairo_pdf)
ggsave("figures/4b-movement-pca-legendtop.pdf", width = 6.83, height = 4.30, dpi = 300, device = cairo_pdf)


movement <- bind_rows (
  read_csv(glue("{results_dir}/wug_wugs_movement_text.csv")) %>%
    mutate(modality='Language'),
  read_csv(glue("{results_dir}/wug_wugs_movement_image.csv")) %>%
    mutate(modality='Vision')
)

# Here is your updated code. The two main changes are explicitly setting the dodge.width to match between the points and the summary, and adding fun.data = "mean_cl_normal" to generate the 95% confidence intervals.

# R
# Make sure the Hmisc package is installed for mean_cl_normal to work!
# install.packages("Hmisc")

movement %>%
  mutate(number = factor(number, levels = c("sg", "pl"))) %>%
  ggplot(aes(number, movement, color = modality, group = modality, shape = modality)) +
  # 1. Explicitly set dodge.width to 0.75 (the standard) so we can match it later
  geom_point(position = position_jitterdodge(dodge.width = 0.75, seed = 1024), alpha = 0.1) +
  # 2. Match the width (0.75) and use mean_cl_normal for 95% CI
  stat_summary(
    fun.data = "mean_cl_normal", 
    geom = "pointrange",
    position = position_dodge(width = 0.75),
    size = 0.4 # Optional: makes the pointrange slightly thicker to stand out
  ) +
  geom_hline(yintercept = 0.0, linetype = "dashed") +
  scale_y_continuous(limits = c(0, 0.2)) +
  theme_classic(base_size = 16, base_family = "Times") +
  labs(
    x = "Number",
    y = "Movement towards\ndesired region",
    color = "Cue Condition",
    shape = "Cue Condition"
  ) +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.95, 0.95),
    legend.justification = c("right", "top")
  )

