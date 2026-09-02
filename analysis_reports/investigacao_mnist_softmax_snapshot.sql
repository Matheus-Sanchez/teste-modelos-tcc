-- Provenance companion for the MCP report artifact.
--
-- The canonical evidence is produced by
-- analysis_reports/investigate_mnist_softmax.py from the saved benchmark
-- artifacts. This portable SQL view exposes the comparison rows used in the
-- report so that the headline result can also be audited in a SQL engine.

WITH comparison(
    activation,
    test_accuracy,
    test_balanced_accuracy,
    test_macro_f1,
    test_loss,
    predicted_class_count,
    predicted_mode_share
) AS (
    VALUES
        ('ReLU',     0.987426176415, 0.987168599752, 0.987190471562, 0.052007563412, 10, 0.112497618594),
        ('Sigmoid',  0.989236044961, 0.989138068751, 0.989162898279, 0.037177644670, 10, 0.112592874833),
        ('Softmax',  0.112497618594, 0.100000000000, 0.020224334275, 2.301202058792,  1, 1.000000000000)
)
SELECT
    activation,
    test_accuracy,
    test_balanced_accuracy,
    test_macro_f1,
    test_loss,
    predicted_class_count,
    predicted_mode_share
FROM comparison
ORDER BY test_macro_f1 DESC;
