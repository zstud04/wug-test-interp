# EACL rebuttal version of the paper's loss-curve figure (Appendix B.2):
# mean training cross-entropy on the novel token per epoch, across seeds, for
# the language-cued runs (the paper's, unchanged) and the NEW vision-cued runs.
# Per-run logs end at each run's best dev epoch (early stopping), so later
# epochs average over fewer seeds; only epochs with at least 10 seeds still
# training are plotted.
#
# On the server: /opt/R/4.6.2/bin/Rscript eacl-rebuttal/analysis/loss-curves-rebuttal.R

library(tidyverse)

read_runs <- function(pattern, model, cue) {
  files <- Sys.glob(pattern)
  map_df(files, function(f) {
    read_csv(f, show_col_types = FALSE, col_select = c(epoch, avg_ce)) %>%
      mutate(run = f)
  }) %>%
    mutate(model = model, cue = cue)
}

runs <- bind_rows(
  read_runs("results/train/CI_seed_runs/lr_ci_results_Qwen3-VL-2B-Instruct_text_lr0p001_50seeds/seed_*/*/epoch_stats.csv",
            "Qwen3-VL-2B", "Language"),
  read_runs("results/train/CI_seed_runs/lr_ci_results_Qwen3-VL-4B-Instruct_text_lr0p001_50seeds/seed_*/*/epoch_stats.csv",
            "Qwen3-VL-4B", "Language"),
  read_runs("eacl-rebuttal/results/lr_search_2B_bs8/init_small/lr_0p0075_joint_bs8/seed_*/*/epoch_stats.csv",
            "Qwen3-VL-2B", "Vision"),
  read_runs("eacl-rebuttal/results/lr_search_2B_bs8_more45/init_small/lr_0p0075_joint_bs8/seed_*/*/epoch_stats.csv",
            "Qwen3-VL-2B", "Vision"),
  read_runs("eacl-rebuttal/results/lr_search_4B_bs8/init_small/lr_0p005_joint_bs8/seed_*/*/epoch_stats.csv",
            "Qwen3-VL-4B", "Vision"),
  read_runs("eacl-rebuttal/results/lr_search_4B_bs8_more45/init_small/lr_0p005_joint_bs8/seed_*/*/epoch_stats.csv",
            "Qwen3-VL-4B", "Vision")
)

runs %>% distinct(model, cue, run) %>% count(model, cue) %>% print()

loss <- runs %>%
  group_by(model, cue, epoch) %>%
  summarize(
    n = n(),
    sd = sd(avg_ce),
    ce = mean(avg_ce),
    conf = qt(0.975, n - 1) * sd / sqrt(n),
    .groups = "drop"
  ) %>%
  filter(n >= 10)

write_csv(loss, "eacl-rebuttal/test_scores/loss_curves_language_vision.csv")

loss %>%
  ggplot(aes(epoch, ce, color = cue, fill = cue)) +
  geom_ribbon(aes(ymin = ce - conf, ymax = ce + conf), color = NA, alpha = 0.3) +
  geom_line() +
  facet_wrap(~model, ncol = 1, axes = "all_x") +
  scale_color_manual(values = c("Language" = "#5e3c99", "Vision" = "#e66101")) +
  scale_fill_manual(values = c("Language" = "#5e3c99", "Vision" = "#e66101")) +
  theme_classic(base_size = 14, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_text(size = 13),
    legend.position = "top",
    legend.title = element_text(size = 13),
    legend.box.spacing = unit(0, "pt"),
    plot.margin = margin(0, 0, 0, 0, "pt"),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.box.background = element_rect(fill = "transparent", color = NA),
    panel.border = element_blank()
  ) +
  guides(
    color = guide_legend(title.position = "top", title.hjust = 0.5),
    fill = guide_legend(title.position = "top", title.hjust = 0.5)
  ) +
  labs(
    x = "Epoch",
    y = "Cross-entropy on novel token",
    color = "Cue Condition",
    fill = "Cue Condition"
  )

ggsave("eacl-rebuttal/figures/loss-curves-language-vision.pdf", width = 3.4, height = 5.6, dpi = 300)
