-- Key business insight queries for Customer Churn & Sales analytics

-- 1) Overall churn rate
SELECT ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
FROM customer_analytics_processed;

-- 2) Churn rate by customer segment
SELECT segment,
       ROUND(AVG(churn) * 100, 2) AS churn_rate_pct,
       COUNT(*) AS customers
FROM customer_analytics_processed
GROUP BY segment
ORDER BY churn_rate_pct DESC;

-- 3) Revenue concentration by region
SELECT region,
       ROUND(SUM(monthly_spend), 2) AS total_monthly_revenue,
       ROUND(AVG(monthly_spend), 2) AS avg_monthly_spend
FROM customer_analytics_processed
GROUP BY region
ORDER BY total_monthly_revenue DESC;

-- 4) High-risk customers to target for retention
SELECT customer_id,
       segment,
       region,
       monthly_spend,
       support_tickets_90d,
       days_since_last_login,
       email_open_rate
FROM customer_analytics_processed
WHERE churn = 1
ORDER BY monthly_spend DESC, support_tickets_90d DESC
LIMIT 25;

-- 5) Monthly sales trend
SELECT month,
       sales,
       LAG(sales) OVER (ORDER BY month) AS previous_month_sales,
       ROUND((sales - LAG(sales) OVER (ORDER BY month))
             / NULLIF(LAG(sales) OVER (ORDER BY month), 0) * 100, 2) AS mom_growth_pct
FROM monthly_sales_processed
ORDER BY month;

-- 6) Top segments by average engagement score
SELECT segment,
       ROUND(AVG(engagement_score), 3) AS avg_engagement,
       ROUND(AVG(monthly_spend), 2) AS avg_spend
FROM customer_analytics_processed
GROUP BY segment
ORDER BY avg_engagement DESC;
