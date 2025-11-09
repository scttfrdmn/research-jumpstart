# Frequently Asked Questions

Common questions about Research Jumpstart, cloud computing for research, and transitioning from traditional workflows.

## Getting Started

??? question "Do I need to know cloud computing to use Research Jumpstart?"
    **No!** That's the whole point. Research Jumpstart is designed to teach you cloud computing through real research projects. Start with [Studio Lab (free)](../getting-started/studio-lab-quickstart.md) and learn by doing.

    **What helps**:
    - Basic Python knowledge
    - Familiarity with Jupyter notebooks
    - Understanding of your research domain

    **What you'll learn**:
    - Cloud computing concepts
    - Working with cloud data
    - Scaling analyses

??? question "How much does it cost?"
    **Studio Lab**: $0 forever (completely free)

    **Unified Studio**: Typical costs
    - Small project: $5-10
    - Medium project: $20-30
    - Large project: $50-100
    - Storage: $5-15/month

    **HPC Hybrid**: $10-20/project

    [Detailed cost calculator →](cost-calculator.md)

??? question "Do I need an AWS account?"
    **For Studio Lab**: No AWS account needed
    **For Unified Studio**: Yes, AWS account required
    **For HPC Hybrid**: Yes, AWS account required

    Setting up an AWS account takes 10-15 minutes and requires a credit card (for verification, not immediate charges).

??? question "Can I use my own data?"
    **Yes!** While Research Jumpstart projects use public datasets for easy access, you can:

    - Upload your data to S3
    - Modify data loading code
    - Adapt projects to your formats

    Each project includes instructions for using custom data.

??? question "Is this only for certain research domains?"
    **No!** Research Jumpstart covers 20+ domains:
    - Climate Science
    - Genomics
    - Medical Research
    - Social Sciences
    - Physics & Astronomy
    - Digital Humanities
    - Economics & Finance
    - And 13+ more...

    [Browse all domains →](../projects/all-domains.md)

---

## Studio Lab

??? question "How long does Studio Lab approval take?"
    **Approval time varies.** Some accounts are approved instantly, others may take several days.

    **Improve your chances of quick approval**:
    - Use institutional (.edu) email if available
    - Be specific and detailed about your research use case
    - Provide affiliation information
    - Clearly articulate your intended projects

??? question "What if Studio Lab rejects my application?"
    Rejections are rare. If rejected:

    1. Reapply with more detail about your research
    2. Use institutional email
    3. Specify concrete research plans
    4. Contact Studio Lab support

??? question "Can I run GPU jobs in Studio Lab?"
    **Yes, but limited.** Studio Lab provides GPU instances (NVIDIA T4) but usage is limited to prevent abuse. Best for:
    - Testing GPU code
    - Small model training
    - Learning GPU programming

    For production GPU work, use Unified Studio.

??? question "What happens when my 12-hour session expires?"
    **Your work is saved!** Notebooks auto-save, and your environment persists.

    **What happens**:
    - Running cells are interrupted
    - Kernel stops
    - Files remain untouched
    - Next login: right where you left off

    **Workaround for long jobs**:
    - Save checkpoints frequently
    - Break into smaller chunks
    - Use Unified Studio for >12hr jobs

??? question "Can I increase Studio Lab storage beyond 15GB?"
    **No.** 15GB is the fixed limit. If you need more:

    **Workarounds**:
    - Delete old projects (save important files first)
    - Use Git for code (don't store in Studio Lab)
    - Process data in chunks
    - Transition to Unified Studio

??? question "Can multiple people share a Studio Lab account?"
    **No, one account per person.** Studio Lab is for individual use.

    **For teams**:
    - Each person gets their own Studio Lab account
    - Share code via Git
    - Coordinate via shared repositories
    - Or use Unified Studio for true collaboration

---

## Unified Studio

??? question "How do I prevent surprise bills?"
    **Set up billing alerts**:

    1. AWS Billing Dashboard → Budgets
    2. Create budget: $50/month (adjust for your needs)
    3. Set alerts at 80% and 100%
    4. Email notifications

    **Additional protection**:
    - Configure auto-shutdown (60min idle timeout)
    - Use spot instances (70% cheaper)
    - Review costs weekly
    - Start small, scale gradually

??? question "What if I accidentally leave an instance running?"
    **It costs money!** But you can minimize damage:

    **Prevention**:
    ```python
    # In notebook config
    {
        "auto_shutdown": True,
        "idle_timeout_minutes": 60
    }
    ```

    **If it happens**:
    1. Stop instance immediately
    2. Check AWS Billing for cost
    3. Set up auto-shutdown
    4. Learn from experience

    **Reality check**: Most "forgot to stop" bills are $20-50, annoying but not catastrophic.

??? question "Can I use Unified Studio without Bedrock?"
    **Yes!** Bedrock is optional. Use Unified Studio for:
    - Distributed computing
    - Large datasets
    - Team collaboration
    - Scale and storage

    Bedrock adds ~$3-5/project if you choose to use it.

??? question "Do I need to learn CloudFormation?"
    **No!** Research Jumpstart provides pre-built CloudFormation templates. Just:

    1. Click "Launch Stack" button
    2. Fill in a few parameters (project name, region)
    3. Wait for deployment (10-15 min)
    4. Start working

    You don't need to write CloudFormation.

??? question "Can I use Unified Studio with my university's AWS account?"
    **Often yes!** Many universities have:
    - Shared AWS accounts for research
    - AWS credits programs
    - Institutional support

    **Check with**:
    - Your research computing group
    - IT department
    - Library (research data services)
    - Lab PI (may have AWS credits)

??? question "What AWS region should I use?"
    **Recommendation: us-west-2 (Oregon)**

    **Why**:
    - Most AWS Open Data is in us-west-2
    - Avoids cross-region transfer costs
    - Good availability

    **Exceptions**:
    - Your institution requires specific region
    - Working with data already in another region
    - Compliance requirements (data residency)

---

## Data & Privacy

??? question "Is my data secure?"
    **Yes**, when following AWS best practices:

    **Security measures**:
    - Data encrypted at rest (S3 encryption)
    - Data encrypted in transit (HTTPS/TLS)
    - Access controls via IAM
    - VPC isolation (if needed)
    - Audit logging (CloudTrail)

    **Your responsibility**:
    - Don't commit secrets to Git
    - Use strong passwords
    - Enable MFA
    - Follow institutional policies

??? question "Can I work with sensitive/protected data?"
    **Yes, with proper configuration.** AWS supports:

    **HIPAA Compliance**:
    - Sign AWS Business Associate Agreement (BAA)
    - Use HIPAA-eligible services
    - Enable encryption
    - Audit logging

    **GDPR Compliance**:
    - Choose appropriate region (EU)
    - Data processing agreements
    - Encryption and access controls

    **Recommendation**: Consult your institution's compliance officer before handling sensitive data.

??? question "Who can see my data in Unified Studio?"
    **Only you (by default).**

    **Access control**:
    - You own the AWS account = you control access
    - Add collaborators via IAM permissions
    - Share specific S3 buckets only
    - Don't make buckets public (unless intended)

    **AWS employees**: Cannot see your data without explicit permission (and legal process).

??? question "What happens to my data if I stop paying AWS?"
    **Data remains in your account but inaccessible.**

    **Best practice**:
    - Before suspending: Download critical data
    - Archive to long-term storage (Glacier)
    - Export final results
    - Document data locations

    **AWS retention**:
    - Data not automatically deleted
    - But you can't access without paying storage costs
    - After months of non-payment, AWS may eventually delete

---

## Technical Questions

??? question "What programming languages can I use?"
    **Research Jumpstart projects use Python** (most common for research).

    **But you can also use**:
    - R (via IRkernel)
    - Julia
    - Bash scripts
    - Any language with Jupyter kernel

    Projects are in Python because:
    - Largest research community
    - Best cloud integration
    - Extensive scientific libraries

??? question "Can I use my favorite Python packages?"
    **Yes!** Install any package:

    ```bash
    # via conda
    conda install package-name

    # via pip
    pip install package-name
    ```

    Projects include `environment.yml` with all required packages.

??? question "How do I debug when things go wrong?"
    **Strategies**:

    1. **Read error messages carefully**
    2. **Check common issues**:
       - Wrong conda environment?
       - Paths correct?
       - Data in expected location?
    3. **Search GitHub issues**: Someone may have hit same problem
    4. **Ask community**: GitHub Discussions
    5. **Use Bedrock** (in Unified Studio): "What does this error mean?"

??? question "Can I schedule jobs to run automatically?"
    **In Unified Studio: Yes!**

    **Options**:
    - AWS Step Functions (workflows)
    - EventBridge (scheduled triggers)
    - Lambda functions (serverless)
    - SageMaker Processing Jobs

    **In Studio Lab: Limited**
    - Can use cron, but session expires after 12 hours
    - Better to use Unified Studio for automation

??? question "How do I work with very large datasets (>1TB)?"
    **Don't download!** Query in place:

    ```python
    # Bad: Download 1TB
    aws s3 cp s3://huge-dataset/ . --recursive  # $90!

    # Good: Query directly
    import s3fs
    fs = s3fs.S3FileSystem()
    data = xarray.open_dataset('s3://huge-dataset/file.nc')
    result = data.sel(time='2020').mean()  # Only result downloads
    ```

    **Also consider**:
    - Distributed processing (Spark/Dask)
    - Partitioned data (parquet, zarr)
    - Incremental processing

---

## Transition & Migration

??? question "When should I transition from Studio Lab to Unified Studio?"
    **Consider transitioning when**:

    ✅ Datasets exceed 10GB
    ✅ Analyses take > 12 hours
    ✅ Need team collaboration
    ✅ Ready for publication work
    ✅ Want AI assistance (Bedrock)
    ✅ Need distributed processing

    **Don't rush!** Many researchers use Studio Lab for 2-4 weeks before transitioning.

??? question "Will I lose my work when I transition?"
    **No!** Use Git to transfer:

    1. **In Studio Lab**: Push to GitHub
    2. **In Unified Studio**: Clone from GitHub
    3. Update data paths (local → S3)
    4. Everything else stays the same

    [Detailed transition guide →](../transition-guides/studio-lab-to-unified.md)

??? question "Can I go back to Studio Lab after using Unified Studio?"
    **Yes!** They're not mutually exclusive:

    - Use Studio Lab for teaching
    - Use Unified Studio for research
    - Use Studio Lab for quick tests
    - Use Unified Studio for production

    Code is portable between both.

??? question "How long does transition take?"
    **Typical timeline**:

    - AWS account setup: 1-2 hours
    - First project migration: 4-6 hours
    - Learning curve: 1-2 weeks

    **Total**: 1-2 weeks to feel comfortable

    After first transition, subsequent projects are much faster.

---

## Collaboration

??? question "How do I collaborate with others?"
    **Options**:

    **Studio Lab** (limited):
    - Share code via Git
    - Share results via email/drive
    - Can't share live environment

    **Unified Studio** (full collaboration):
    - Shared projects
    - Multiple users in same environment
    - Shared S3 buckets
    - Team dashboards

??? question "Can I share my analysis with non-technical stakeholders?"
    **Yes!** Several options:

    1. **Static reports**: Export notebook to PDF/HTML
    2. **Dashboards**: Create web dashboard (Streamlit, Dash)
    3. **Generated reports**: Use Bedrock to create policy briefs
    4. **Shared S3 links**: Presigned URLs (time-limited access)

??? question "Can my advisor/collaborator review my work without AWS?"
    **Yes!**

    **For code review**:
    - Push to GitHub
    - Share repository link
    - They can view without running

    **For results**:
    - Export figures/tables
    - Generate PDF report
    - Share via normal methods

    **For interactive access**:
    - They need AWS account (or use your shared environment)

---

## HPC Hybrid

??? question "Do I still need my campus HPC if I use cloud?"
    **Not necessarily, but HPC can complement cloud:**

    **Keep HPC for**:
    - Heavy compute (if free/subsidized)
    - Workflows already working well
    - Institutional requirements

    **Use cloud for**:
    - Analysis and visualization
    - Collaboration
    - Long-term storage
    - Sharing results

??? question "How do I transfer data between HPC and cloud?"
    **Use AWS CLI**:

    ```bash
    # On HPC, after job completes
    module load aws-cli
    aws s3 sync /scratch/results/ s3://my-bucket/results/
    ```

    **Tips**:
    - Transfer overnight (large files)
    - Compress before transfer
    - Use parallel uploads
    - Check institutional data transfer policies

??? question "Is hybrid approach worth the complexity?"
    **Depends on your situation**:

    **Yes, if**:
    - Good HPC access (no queues)
    - Cost-conscious
    - Gradual cloud adoption preferred
    - Institutional cloud policies restrictive

    **No, if**:
    - Poor HPC access (long queues)
    - Time is more valuable than cost
    - Want simplicity
    - No HPC access at all

---

## Troubleshooting

??? question "Why is my notebook running slowly?"
    **Possible causes**:

    **In Studio Lab**:
    - Shared resources (peak times)
    - Inefficient code (loops instead of vectorized)
    - Loading too much data

    **In Unified Studio**:
    - Instance too small (upgrade instance type)
    - Network bottleneck (use same region as data)
    - Inefficient queries (optimize data access)

??? question "I got an 'out of memory' error. What now?"
    **Solutions**:

    1. **Process in chunks**:
       ```python
       # Instead of loading all at once
       for chunk in pd.read_csv('large.csv', chunksize=10000):
           process(chunk)
       ```

    2. **Upgrade instance** (Unified Studio):
       - Move from m5.xlarge to m5.2xlarge
       - Doubles RAM

    3. **Use Dask/Spark**:
       - Distributed processing
       - Handles larger-than-memory data

??? question "My AWS costs are higher than expected. Why?"
    **Common culprits**:

    1. **Forgot to stop instance** (most common!)
    2. **Data transfer** (moved data between regions)
    3. **On-demand instead of spot** (3x more expensive)
    4. **Oversized instance** (using bigger than needed)

    **Investigation**:
    - AWS Cost Explorer → See breakdown
    - Tag resources → Track by project
    - Enable detailed billing

??? question "How do I get help if I'm stuck?"
    **Resources in order**:

    1. **Project README**: Check troubleshooting section
    2. **Search GitHub Issues**: Someone may have hit same problem
    3. **Ask in Discussions**: Community often responds in hours
    4. **Office Hours**: Live help Tuesdays 2-4pm PT
    5. **Email**: hello@researchjumpstart.org

    **When asking for help**:
    - Describe what you're trying to do
    - Share error message (full text)
    - Mention which project/platform
    - What you've tried already

---

## Contributing

??? question "How can I contribute a project?"
    **We'd love your project!**

    **Requirements**:
    - Working Studio Lab version (free tier)
    - Working Unified Studio version (production)
    - Comprehensive documentation
    - Follows our template

    [Detailed contribution guide →](../CONTRIBUTING.md)

??? question "I found a bug. How do I report it?"
    **Thank you!**

    1. [Open an issue on GitHub](https://github.com/research-jumpstart/research-jumpstart/issues)
    2. Use the bug report template
    3. Include:
       - What you expected
       - What actually happened
       - Error messages
       - Steps to reproduce

??? question "Can I translate documentation?"
    **Yes!** Translations are welcome. Open an issue to discuss which language and we'll coordinate.

---

## Still Have Questions?

- 💬 [GitHub Discussions](https://github.com/research-jumpstart/research-jumpstart/discussions) - Ask the community
- 📧 [Email](mailto:hello@researchjumpstart.org) - Contact us directly
- 📅 [Office Hours](../community/office-hours.md) - Live help every Tuesday
- 🎥 [Video Tutorials](videos.md) - Watch walkthroughs

**Not finding your question?** [Open a discussion](https://github.com/research-jumpstart/research-jumpstart/discussions/new) and ask!
