library(tidyverse)

natural_results <- fs::dir_ls("results/eval/attractors/", regexp = "*target_natural_scored.csv", recurse = TRUE) %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    params = case_when(
      str_detect(file, "2B") ~ "Qwen3-VL-2B",
      TRUE ~ "Qwen3-VL-4B"
    )
  ) %>%
  select(-file)

all_seed_results <- fs::dir_ls("results/eval/attractors-50seeds", regexp = "attractor_scored*", recurse = TRUE) %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    params = case_when(
      str_detect(file, "2B") ~ "Qwen3-VL-2B",
      TRUE ~ "Qwen3-VL-4B"
    ),
    source = case_when(
      str_detect(file, "image") ~ "vision",
      TRUE ~ "language"
    )
  ) %>%
  select(-file)

best_seed_results <- fs::dir_ls("results/eval/attractors/", regexp = "*target_wug", recurse = TRUE) %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    params = case_when(
      str_detect(file, "2B") ~ "Qwen3-VL-2B",
      TRUE ~ "Qwen3-VL-4B"
    ),
    source = case_when(
      str_detect(file, "syntax") ~ "language",
      TRUE ~ "vision"
    )
  ) %>%
  select(-file)

best_seed_results %>%
  select(params, source, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all) %>%
  pivot_longer(is_correct_singular:is_correct_all, names_to = "correctness", values_to = "correct") %>%
  group_by(params, source, correctness, attractors) %>%
  summarize(
    acc = mean(correct)
  ) 

natural_results %>%
  select(params, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all) %>%
  pivot_longer(is_correct_singular:is_correct_all, names_to = "correctness", values_to = "correct") %>%
  group_by(params, correctness, attractors) %>%
  summarize(
    acc = mean(correct)
  )

best_seed_results %>%
  select(params, source, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all) %>%
  pivot_longer(is_correct_singular:is_correct_all, names_to = "correctness", values_to = "correct") %>%
  filter(correctness != "is_correct_all") %>%
  group_by(params, source, attractors) %>%
  summarize(
    acc = mean(correct)
  ) %>%
  ggplot(aes(attractors, acc, color = source)) +
  geom_point() +
  geom_line() + 
  facet_wrap(~params) +
  scale_y_continuous(limits = c(0.5, 1))

natural_acc <- natural_results %>%
  select(params, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all) %>%
  pivot_longer(is_correct_singular:is_correct_all, names_to = "correctness", values_to = "correct") %>%
  filter(correctness != "is_correct_all") %>%
  group_by(params, attractors) %>%
  summarize(
    acc = mean(correct),
    .groups = "drop"
  ) 
# 
# natural_acc_source <- bind_rows(
#   natural_acc %>% mutate(source = "language"),
#   natural_acc %>% mutate(source = "vision")
# )

all_seed_acc <- all_seed_results %>%
  select(params, source, seed, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all) %>%
  pivot_longer(is_correct_singular:is_correct_all, names_to = "correctness", values_to = "correct") %>%
  filter(correctness != "is_correct_all") %>%
  group_by(params, source, attractors, seed) %>%
  summarize(
    acc = mean(correct)
  ) %>%
  ungroup() %>%
  group_by(params, source, attractors) %>%
  summarize(
    n = n(),
    sd = sd(acc),
    conf = qt(1 - (0.05/2), n - 1) * sd/sqrt(n),
    acc = mean(acc),
    .groups = "drop"
  )

bind_rows(
  natural_acc %>% mutate(source = "language", type = "real"),
  natural_acc %>% mutate(source = "vision", type = "real"),
  all_seed_acc %>% mutate(type = "novel")
) %>%
  ggplot(aes(attractors, acc, shape = type, linetype = type, color = source, fill = source)) +
  geom_point() +
  geom_line() + 
  geom_ribbon(aes(ymin = acc-conf, ymax = acc+conf), color = NA, alpha = 0.4) +
  facet_wrap(~params) +
  scale_y_continuous(limits = c(0.5, 1)) +
  labs(
    color = "Cue Condition",
    fill = "Cue Condition",
    shape = "Word Type",
    linetype = "Word Type"
  )

bind_rows(
  natural_acc %>% mutate(source = "language", type = "Real"),
  natural_acc %>% mutate(source = "vision", type = "Real"),
  all_seed_acc %>% mutate(type = "Novel")
) %>%
  mutate(color_group = ifelse(type == "Real", "Pre-training", source)) %>% 
  mutate(
    source = str_to_title(source),
    color_group = str_to_title(color_group)
  ) %>%
  ggplot(aes(attractors, acc, 
             shape = type, linetype = type, 
             color = color_group, fill = color_group)) +
  geom_point() +
  geom_line() + 
  geom_ribbon(aes(ymin = acc-conf, ymax = acc+conf), color = NA, alpha = 0.4) +
  facet_wrap(~params, scales = "free") +
  scale_y_continuous(limits = c(0.5, 1), labels = scales::percent_format(suffix = "")) +
  scale_color_manual(
    name = "Cue Condition",
    values = c("Pre-Training" = "black", "Language" = "#5e3c99", "Vision" = "#e66101"),
    breaks = c("Language", "Vision") # This line hides "pre-training" from the legend
  ) +
  scale_fill_manual(
    name = "Cue Condition",
    values = c("Pre-Training" = "black", "Language" = "#5e3c99", "Vision" = "#e66101"),
    breaks = c("Language", "Vision") # This line hides "pre-training" from the legend
  ) +
  labs(
    shape = "Word Type",
    linetype = "Word Type"
  ) +
  guides(
    # Add title.position = "top" to all guides
    color = guide_legend(title.position = "top", title.hjust = 0.5),
    fill = guide_legend(title.position = "top", title.hjust = 0.5),
    shape = guide_legend(title.position = "top", title.hjust = 0.5, override.aes = list(fill = NA)),
    linetype = guide_legend(title.position = "top", title.hjust = 0.5, override.aes = list(fill = NA))
  ) +
  # guides(
  #   shape = guide_legend(override.aes = list(fill = NA)),
  #   linetype = guide_legend(override.aes = list(fill = NA))
  # ) + 
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_text(size = 14),
    legend.position = "top",
    legend.title = element_text(size = 14),
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
    x = "Attractors",
    y = "Accuracy (%)"
  )

# ggsave("figures/behavioral-accuracies-attractors.pdf", width = 7.14, height = 3.07, dpi = 300, device=cairo_pdf)
ggsave("figures/behavioral-accuracies-attractors-legendtop.pdf", width = 5, height = 3.45, dpi = 300)
