# Learning Analytics Platform with AWS - Tier 2

**Duration:** 2-4 hours | **Cost:** $6-11 | **Platform:** AWS (boto3, no CloudFormation)

Build a serverless learning analytics pipeline to analyze student performance, identify at-risk students, and generate actionable insights using AWS services.

## Project Overview

This Tier 2 project demonstrates how to build a cloud-based learning analytics platform using AWS services. You'll upload student activity data to S3, deploy Lambda functions to analyze learning patterns, store results in DynamoDB, and query educational insights using Athena.

**Key Learning:** Move from local data analysis (Tier 1) to scalable cloud-based educational data mining (Tier 2)

## What You'll Build

A complete learning analytics pipeline that:
- Uploads student activity data (quiz scores, assignments, engagement) to AWS S3
- Processes learning analytics with AWS Lambda (grade analysis, at-risk identification)
- Stores student performance metrics in DynamoDB
- Queries educational insights with Athena (class-level statistics)
- Generates actionable recommendations for educators
- Protects student privacy through data anonymization

## Prerequisites

### Required
- AWS Account (with free tier eligible)
- Python 3.8+
- Jupyter Notebook or JupyterLab
- AWS CLI configured with credentials
- boto3 library

### Knowledge
- Basic Python programming
- Familiarity with educational data concepts (grades, assessments)
- Understanding of learning analytics basics (helpful but not required)
- Basic SQL for Athena queries (optional)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              Learning Analytics Platform                     │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│ Student Activity Data│
│ - Quiz Scores        │
│ - Assignments        │
│ - Engagement Logs    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ upload_to_s3.py      │
│ (Anonymize & Upload) │
└──────────┬───────────┘
           │
           ▼
    ┌──────────────┐
    │   AWS S3     │
    │  Raw Data    │  ◄─────────────────┐
    │ CSV Files    │                     │
    └──────┬───────┘                     │
           │                             │
           │ S3 Event Trigger            │
           ▼                             │
    ┌──────────────────────┐             │
    │  AWS Lambda Function │             │
    │ analyze-performance  │             │
    │  - Grade averages    │             │
    │  - Completion rates  │             │
    │  - At-risk detection │             │
    │  - Learning curves   │             │
    │  - Mastery metrics   │             │
    └──────────┬───────────┘             │
               │                         │
               ▼                         │
        ┌──────────────┐                 │
        │  AWS S3      │                 │
        │ Results      │─────────────────┤
        │ (Parquet)    │                 │
        └──────────────┘                 │
               │                         │
               ▼                         │
        ┌──────────────┐                 │
        │ AWS DynamoDB │                 │
        │StudentMetrics│─────────────────┤
        │ - Student ID │                 │
        │ - Avg Grade  │                 │
        │ - Risk Level │                 │
        │ - Engagement │                 │
        └──────────────┘                 │
               │                         │
               ▼                         │
        ┌──────────────┐                 │
        │ AWS Athena   │                 │
        │ SQL Queries  │─────────────────┤
        │ - Class stats│                 │
        │ - Trends     │                 │
        └──────────────┘                 │
               │                         │
    ┌──────────▼──────────┐              │
    │ query_results.py    │              │
    │ (Retrieve & Analyze)│──────────────┘
    └─────────────────────┘

        ┌─────────────────┐
        │  Jupyter        │
        │  Notebook       │
        │  - Dashboards   │
        │  - Visualizations│
        │  - Predictions  │
        └─────────────────┘
```

## AWS Services Used

| Service | Purpose | Cost |
|---------|---------|------|
| **S3** | Store student data and analytics results | $0.023 per GB/month (~$0.1-0.3) |
| **Lambda** | Serverless data processing and analytics | $0.20 per 1M invocations (~$0.1-0.2) |
| **DynamoDB** | NoSQL storage for student metrics | $0.25/GB for on-demand (~$0.2-0.5) |
| **Athena** | SQL queries on S3 data | $5 per TB scanned (~$0.05-0.1) |
| **IAM** | Access control and permissions | Free |
| **CloudWatch** | Logging and monitoring | Free (small amounts) |

**Total Estimated Cost: $6-11 per run**

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS

Follow the detailed setup guide:
```bash
cat setup_guide.md
```

Quick steps:
- Create S3 bucket: `learning-analytics-{your-user-id}`
- Create IAM role for Lambda
- Create DynamoDB table: `StudentAnalytics`
- Set up Athena workspace

### 3. Run the Pipeline

```bash
# Upload sample student data
python scripts/upload_to_s3.py

# Deploy Lambda function (see setup_guide.md for deployment steps)

# Query results
python scripts/query_results.py

# Analyze in notebook
jupyter notebook notebooks/learning_analysis.ipynb
```

## Project Structure

```
tier-2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_guide.md                     # Step-by-step AWS setup
├── cleanup_guide.md                   # How to delete resources
├── notebooks/
│   └── learning_analysis.ipynb       # Main analysis notebook
└── scripts/
    ├── upload_to_s3.py               # Upload student data to S3
    ├── lambda_function.py             # Lambda analytics code
    └── query_results.py               # Retrieve and analyze results
```

## Workflow Steps

### Step 1: Setup AWS Environment
- Create S3 bucket for student data
- Create IAM role for Lambda
- Create DynamoDB table for student metrics
- Set up Athena workspace
- (Detailed instructions in `setup_guide.md`)

### Step 2: Prepare Student Data
- Generate or use provided sample student data
- Data includes: student IDs, quiz scores, assignments, engagement metrics
- Format: CSV files with timestamp, student_id, activity_type, score, etc.

### Step 3: Upload Data to S3
```bash
python scripts/upload_to_s3.py --input-dir ./sample_data \
                                --s3-bucket learning-analytics-{user-id} \
                                --prefix raw-data/
```

### Step 4: Deploy Lambda Function
- Package `lambda_function.py` with dependencies
- Deploy to AWS Lambda
- Configure S3 event trigger
- Test with sample invocation
- (Detailed instructions in `setup_guide.md`)

### Step 5: Process Student Data
- Lambda automatically triggers on S3 uploads
- Or manually invoke Lambda for batch processing
- Lambda performs analytics:
  - Calculate grade averages (overall, by course, by assessment type)
  - Compute completion rates
  - Calculate engagement scores (time on task, resource access)
  - Identify at-risk students (low grades, declining performance)
  - Analyze learning curves (improvement over time)
  - Compute mastery learning metrics

### Step 6: Store Metrics in DynamoDB
- Lambda logs student metrics to DynamoDB:
  - Student ID (anonymized)
  - Average grade, median grade, grade trend
  - Completion rate (assignments submitted / assigned)
  - Engagement score (normalized 0-100)
  - Risk level (low, medium, high)
  - Last updated timestamp
  - Course and cohort information

### Step 7: Query Results
```bash
python scripts/query_results.py --table-name StudentAnalytics \
                                 --risk-level high \
                                 --limit 50
```

### Step 8: Analyze with Athena
- Create Athena table from S3 results
- Run SQL queries for class-level insights:
  - Average class performance
  - Grade distributions
  - Performance trends over time
  - Correlation between engagement and grades

### Step 9: Visualize in Notebook
- Open `notebooks/learning_analysis.ipynb`
- Query DynamoDB for student metrics
- Generate learning analytics dashboards
- Create at-risk student reports
- Visualize grade distributions and trends
- Analyze intervention effectiveness

## Expected Results

After completing this project, you will have:

1. **Educational Data Mining Experience**
   - Uploaded and anonymized student data in S3
   - Processed learning activity logs at scale
   - Generated actionable educational insights
   - Protected student privacy

2. **Learning Analytics Pipeline**
   - Deployed serverless analytics functions
   - Calculated comprehensive student metrics
   - Identified at-risk students early
   - Generated educator recommendations

3. **NoSQL Data Management**
   - Stored structured student metrics in DynamoDB
   - Queried performance data efficiently
   - Understood on-demand vs provisioned capacity

4. **SQL Analytics with Athena**
   - Created external tables on S3 data
   - Ran SQL queries for class-level insights
   - Analyzed large datasets without data movement

5. **Reproducible Analytics Pipeline**
   - Created reusable Python scripts
   - Documented setup and execution
   - Prepared foundation for production deployment

## Learning Analytics Metrics

### Grade Analysis
- **Average Grade:** Mean score across all assessments
- **Median Grade:** Middle score (robust to outliers)
- **Grade Trend:** Linear regression slope of grades over time
- **Standard Deviation:** Consistency of performance

### Completion Metrics
- **Completion Rate:** Percentage of assignments submitted
- **On-Time Rate:** Percentage submitted before deadline
- **Late Submission Rate:** Percentage submitted late

### Engagement Metrics
- **Time on Task:** Total time spent on learning activities
- **Resource Access:** Frequency of accessing course materials
- **Forum Participation:** Posts, replies, questions asked
- **Login Frequency:** Days active per week

### At-Risk Indicators
- **Low Grade:** Below 70% average
- **Declining Performance:** Negative grade trend
- **Low Completion:** <80% completion rate
- **Low Engagement:** <50% engagement score
- **Multiple Factors:** Combined risk score

### Mastery Learning
- **Learning Gain:** Difference between first and last assessment
- **Mastery Level:** Percentage of learning objectives achieved
- **Learning Velocity:** Rate of improvement over time
- **Retention Rate:** Performance on cumulative assessments

## Cost Breakdown

### Detailed Cost Estimate

**Assumptions:**
- 500 students in dataset
- 10 assessments per student (5,000 records)
- 3 courses tracked
- Data size: ~5 MB raw CSV
- Lambda: 50 invocations, 20 seconds each
- Athena: 5 queries scanning 50 MB each
- Run duration: 1 week

**Costs:**

| Service | Usage | Cost |
|---------|-------|------|
| S3 Storage | 10 MB for 1 week | $0.000003 per GB-hour = $0.0002 |
| S3 Uploads | 50 PUT requests | $0.005 per 1000 = $0.0003 |
| S3 Downloads | 20 GET requests | $0.0004 per 1000 = $0.0001 |
| Lambda Compute | 50 × 20s @ 512MB | ~$0.08 |
| Lambda Requests | 50 requests | ~$0.00001 |
| DynamoDB Writes | 500 writes on-demand | $1.25 per 1M = $0.0006 |
| DynamoDB Reads | 100 reads on-demand | $0.25 per 1M = $0.00003 |
| DynamoDB Storage | 1 MB for 1 week | $0.25 per GB-month = $0.001 |
| Athena Queries | 5 queries × 50 MB | $5 per TB = $0.0003 |
| Data Transfer | ~10 MB out | $0.12 per GB = $0.001 |
| **Total** | | **$0.08** |

**Note:** This is a minimal run. Costs increase with:
- More students: +$0.001 per 100 students
- More assessments: +$0.005 per 1000 records
- More Lambda processing: +$0.002 per 100 Lambda seconds
- More Athena queries: +$0.005 per 1 GB scanned

**Typical range for learning project: $6-11**

This includes safety margin, multiple test runs, and larger datasets for realistic learning scenarios.

## Educational Use Cases

### For Instructors
- **Early Intervention:** Identify struggling students before it's too late
- **Personalized Support:** Target interventions based on specific needs
- **Course Design:** Analyze which assessments are most predictive
- **Engagement Tracking:** Monitor student participation patterns

### For Administrators
- **Program Assessment:** Evaluate overall program effectiveness
- **Resource Allocation:** Identify courses needing additional support
- **Retention Analysis:** Predict and prevent student dropout
- **Outcome Tracking:** Measure learning outcomes across cohorts

### For Researchers
- **Learning Science:** Test hypotheses about effective pedagogy
- **Predictive Modeling:** Build models for student success
- **Intervention Studies:** Evaluate effectiveness of interventions
- **Educational Data Mining:** Discover patterns in learning data

## Learning Objectives

### Technical Skills
- [x] Create and configure AWS services (S3, Lambda, DynamoDB, Athena, IAM)
- [x] Write boto3 code to interact with AWS services
- [x] Deploy Lambda functions with external dependencies
- [x] Design NoSQL data models for educational data
- [x] Write SQL queries with Athena
- [x] Monitor costs and optimize spending

### AWS Concepts
- [x] Object storage and S3 lifecycle policies
- [x] Serverless computing with Lambda
- [x] Event-driven architectures (S3 triggers)
- [x] NoSQL databases (DynamoDB)
- [x] Serverless SQL queries (Athena)
- [x] IAM roles and permissions
- [x] Pay-per-use cloud pricing

### Learning Analytics
- [x] Student performance metrics
- [x] At-risk student identification
- [x] Engagement scoring
- [x] Mastery learning concepts
- [x] Learning curve analysis
- [x] Data privacy and anonymization

## Privacy and Ethics

### Data Anonymization
- Student IDs are hashed before upload
- Personally identifiable information (PII) removed
- Only academic performance data retained
- Compliance with FERPA guidelines

### Ethical Considerations
- Use for student support, not punishment
- Transparent about data collection
- Secure storage and access control
- Regular data retention review

### Best Practices
- Minimize data collection to essential metrics
- Implement role-based access control
- Encrypt data at rest and in transit
- Document data handling procedures

## Troubleshooting

### Common Issues

**Problem:** "NoCredentialsError" when running Python scripts
- **Solution:** Configure AWS CLI with `aws configure` and provide access keys
- See setup_guide.md Step 1 for detailed instructions

**Problem:** Lambda function timeout
- **Solution:** Increase timeout in Lambda configuration to 60 seconds
- Check processing time in CloudWatch logs
- Consider batch processing for large datasets

**Problem:** High S3 costs
- **Solution:** Delete resources with cleanup_guide.md when done
- Use S3 Intelligent-Tiering for long-term storage
- Set lifecycle policies to delete old data

**Problem:** "AccessDenied" errors
- **Solution:** Check IAM role permissions in setup_guide.md
- Ensure Lambda execution role has S3, DynamoDB, and CloudWatch permissions

**Problem:** DynamoDB read/write throttling
- **Solution:** Use on-demand billing (auto-scales)
- Or increase provisioned capacity for predictable workloads

**Problem:** Athena query fails
- **Solution:** Check S3 path in CREATE TABLE statement
- Ensure data format matches table schema
- Verify query results bucket exists

**Problem:** CSV parsing errors in Lambda
- **Solution:** Validate CSV format (headers, delimiters)
- Check for special characters in data
- Handle missing values appropriately

See `cleanup_guide.md` for information on removing test resources.

## Next Steps

After completing this Tier 2 project:

### Option 1: Advanced Tier 2 Features
- Add SNS email notifications for at-risk student alerts
- Implement SQS queue for handling bulk data uploads
- Add CloudWatch dashboard for real-time monitoring
- Implement predictive models with scikit-learn in Lambda
- Create automated weekly reports

### Option 2: Move to Tier 3 (Production)
Tier 3 uses CloudFormation for automated infrastructure:
- Infrastructure-as-code templates
- One-click deployment for institutions
- Multi-tenant support (multiple schools/courses)
- Production-ready security policies
- Auto-scaling and high availability
- Integration with LMS (Canvas, Moodle, Blackboard)

See `/projects/education/learning-analytics-platform/tier-3/` for production deployment.

### Option 3: Integrate Real Learning Data
- Connect to LMS APIs (Canvas, Moodle)
- Process real student activity logs
- Implement additional learning analytics metrics
- Build predictive dropout models
- Create instructor dashboards

### Option 4: Research Applications
- Test educational interventions
- Analyze effectiveness of different teaching methods
- Study correlation between engagement and outcomes
- Publish findings in learning analytics conferences

## References

### AWS Documentation
- [S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/)
- [Athena Documentation](https://docs.aws.amazon.com/athena/)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

### Learning Analytics
- [Society for Learning Analytics Research (SoLAR)](https://www.solaresearch.org/)
- [Learning Analytics Review](https://epubs.scu.edu.au/cgi/viewcontent.cgi?article=1058&context=hahs_pubs)
- [Educational Data Mining](https://educationaldatamining.org/)
- [FERPA Privacy Guidelines](https://www2.ed.gov/policy/gen/guid/fpco/ferpa/index.html)

### Educational Metrics
- [Predictive Models in Student Success](https://er.educause.edu/articles/2018/4/predictive-analytics-in-higher-education)
- [At-Risk Student Identification](https://www.sciencedirect.com/science/article/pii/S0747563214002611)
- [Engagement Metrics](https://www.educause.edu/research-and-publications/books/learning-analytics)

### AWS Best Practices
- [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/)
- [AWS Cost Optimization](https://aws.amazon.com/blogs/aws-cost-management/)
- [Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)

## Support

### Getting Help
1. Check troubleshooting section above
2. Review AWS service error messages in CloudWatch logs
3. Consult boto3 documentation for API details
4. Check AWS service quotas (may need increase for scale)

### For Issues
- Review setup_guide.md for configuration problems
- Check IAM permissions if access errors occur
- Verify S3 bucket naming (globally unique)
- Confirm DynamoDB table exists before running scripts
- Validate Athena query syntax

## License

This project is part of the Research Jumpstart curriculum and is provided for educational purposes.

## Author Notes

This is a Tier 2 (AWS Starter) project. It bridges Tier 1 (Studio Lab free tier) and Tier 3 (Production CloudFormation).

**Time to Complete:** 2-4 hours
**Cost:** $6-11 per run
**Difficulty:** Intermediate (requires AWS account setup)

For questions about the project structure, see `TIER_2_SPECIFICATIONS.md` in the project root.

## Acknowledgments

This project demonstrates educational data analytics patterns commonly used in:
- Higher education institutions
- K-12 schools with digital learning platforms
- Online course providers (MOOCs)
- Corporate training programs

The analytics approaches are based on research from the learning analytics and educational data mining communities.
