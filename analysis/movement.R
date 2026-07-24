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

wug_pca %>%
  pivot_wider(names_from = stage, values_from = c(x, y)) %>%
  ggplot() + 
  geom_point(data = real_nouns, aes(x, y, color = number), alpha = 0.08) +
  scale_color_manual(name = "Known Noun\nType", values = c("#018571", "#a6611a")) +
  guides(color = guide_legend(override.aes = list(alpha = 1))) +
  new_scale_color() +
  geom_segment(
    aes(
      x = x_init, y = y_init,
      xend = x_final, yend = y_final,
      color = type
    ), 
    arrow = arrow(length = unit(0.1, "cm")),
    linewidth = 0.6, alpha = 0.6
  ) +
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
    family = "Helvetica",
    fontface = "italic",
    size = 4
  ) +
  geom_text(
    data = data.frame(modality="Language"),
    aes(x = 0.33, y = 0.3),
    label = "final",
    family = "Helvetica",
    fontface = "italic",
    size = 4
  ) +
  scale_color_manual(name = "Novel word", values = rev(c("#018571", "#a6611a"))) +
  guides(color = guide_legend(override.aes = list(alpha = 1))) +
  facet_wrap(~modality, scales="free") + 
  # theme_minimal(base_size = 17, base_family = "Times") +
  theme_classic(base_size = 17, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_text(face='bold.italic')
  ) +
  labs(
    x = "PC1",
    y = "PC2"
  ) 


movement <- bind_rows (
  read_csv(glue("{results_dir}/wug_wugs_movement_text.csv")) %>%
    mutate(modality='Language'),
  read_csv(glue("{results_dir}/wug_wugs_movement_image.csv")) %>%
    mutate(modality='Vision')
)

movement %>%
  mutate(number = factor(number, levels = c("sg", "pl"), labels = c("SG", "PL"))) %>%
  ggplot(aes(number, movement, color = modality, group = modality, shape = modality)) +
  geom_point(position = position_jitterdodge(seed = 1024), alpha = 0.6) +
  geom_hline(yintercept = 0.0, linetype = "dashed") +
  scale_y_continuous(limits = c(0, 0.2)) +
  theme_classic(base_size = 17, base_family = "Times") +
  labs(
    x = "Number",
    y = "Movement towards\ndesired region",
    color = "Input Modality",
    shape = "Input Modality",
  ) +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.95, 0.95),
    legend.justification = c("right", "top")
  )

