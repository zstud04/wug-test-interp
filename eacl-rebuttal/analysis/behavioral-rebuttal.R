# EACL rebuttal version of the paper's Figure 2 (analysis/behavioral.R ->
# figures/behavioral-accuracies-attractors-legendtop.pdf): same code, size and
# style. Real-noun and language-cued (50-seed) results are the paper's,
# unchanged; the vision-cued results are the NEW embeddings (matched captions,
# held-out-image dev, 50 seeds), read from the git-tracked compact test scores
# in eacl-rebuttal/test_scores/ (scripts/build_test_score_tables.py).
#
# please open the .Rproj file in the home dir of the repo so that the paths resolve.
# On the server: /opt/R/4.6.2/bin/Rscript eacl-rebuttal/analysis/behavioral-rebuttal.R

library(tidyverse)

natural_results <- fs::dir_ls("results/eval/attractors/", regexp = "*target_natural_scored.csv", recurse = TRUE) %>%
  map_df(read_csv, .id = "file", show_col_types = FALSE) %>%
  mutate(
    params = case_when(
      str_detect(file, "2B") ~ "Qwen3-VL-2B",
      TRUE ~ "Qwen3-VL-4B"
    )
  ) %>%
  select(-file)

# Language-cued: the paper's 50-seed results, unchanged.
language_results <- fs::dir_ls("results/eval/attractors-50seeds", regexp = "attractor_scored_text", recurse = TRUE) %>%
  map_df(read_csv, .id = "file", show_col_types = FALSE) %>%
  mutate(
    params = case_when(
      str_detect(file, "2B") ~ "Qwen3-VL-2B",
      TRUE ~ "Qwen3-VL-4B"
    ),
    source = "language"
  ) %>%
  select(params, source, seed, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all)

# Vision-cued: new embeddings. Correctness as in core/eval/embed_eval.py --paired (good > bad).
vision_results <- bind_rows(
  read_csv("eacl-rebuttal/test_scores/scores_qwen3-vl-2b.csv", show_col_types = FALSE) %>% mutate(params = "Qwen3-VL-2B"),
  read_csv("eacl-rebuttal/test_scores/scores_qwen3-vl-4b.csv", show_col_types = FALSE) %>% mutate(params = "Qwen3-VL-4B")
) %>%
  mutate(
    source = "vision",
    is_correct_singular = score_good_singular > score_bad_singular,
    is_correct_plural = score_good_plural > score_bad_plural,
    is_correct_all = is_correct_singular & is_correct_plural
  ) %>%
  select(params, source, seed, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all)

all_seed_results <- bind_rows(language_results, vision_results)

natural_acc <- natural_results %>%
  select(params, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all) %>%
  pivot_longer(is_correct_singular:is_correct_all, names_to = "correctness", values_to = "correct") %>%
  filter(correctness != "is_correct_all") %>%
  group_by(params, attractors) %>%
  summarize(
    acc = mean(correct),
    .groups = "drop"
  )

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

print(all_seed_acc, n = Inf)
write_csv(all_seed_acc, "eacl-rebuttal/test_scores/behavioral_accuracy_language_vision.csv")

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
    color = guide_legend(title.position = "top", title.hjust = 0.5),
    fill = guide_legend(title.position = "top", title.hjust = 0.5),
    shape = guide_legend(title.position = "top", title.hjust = 0.5, override.aes = list(fill = NA)),
    linetype = guide_legend(title.position = "top", title.hjust = 0.5, override.aes = list(fill = NA))
  ) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_text(size = 14),
    legend.position = "top",
    legend.title = element_text(size = 14),
    legend.box.spacing = unit(0, "pt"),
    plot.margin = margin(0, 0, 0, 0, "pt"),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.box.background = element_rect(fill = "transparent", color = NA),
    panel.border = element_blank()
  ) +
  labs(
    x = "Attractors",
    y = "Accuracy (%)"
  )

ggsave("eacl-rebuttal/figures/behavioral-accuracies-attractors-legendtop.pdf", width = 5, height = 3.45, dpi = 300)
