#!/usr/bin/env python3
"""
Presentation Demo Preparation Script
Generates all visual assets and tests demo commands for the NHL xG presentation
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add data_pipeline/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_pipeline', 'src'))

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🎬 {title}")
    print(f"{'='*60}")

def run_command_safely(command, description):
    """Run a command safely with error handling."""
    print(f"\n🔧 {description}")
    print(f"Command: {command}")
    print("-" * 40)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output preview:")
                print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print("❌ FAILED")
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT - Command took too long")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False

def check_data_availability():
    """Check if the NHL database is available."""
    print_section("DATA AVAILABILITY CHECK")
    
    db_path = "nhl_stats.db"
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"✅ Database found: {db_path} ({size_mb:.1f} MB)")
        
        # Quick data check
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            
            # Check games
            games_count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            print(f"📊 Games in database: {games_count:,}")
            
            # Check events
            events_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            print(f"📊 Events in database: {events_count:,}")
            
            # Check shots
            shots_count = conn.execute("""
                SELECT COUNT(*) FROM events 
                WHERE eventType IN ('goal', 'shot-on-goal')
            """).fetchone()[0]
            print(f"🏒 Shot events: {shots_count:,}")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
    else:
        print(f"❌ Database not found: {db_path}")
        print("💡 Run data collection first: python ../data_pipeline/src/data/scrape_nhl_data.py")
        return False

def test_core_functionality():
    """Test core analysis functionality."""
    print_section("CORE FUNCTIONALITY TEST")
    
    # Test basic analysis
    success1 = run_command_safely(
        "python ../data_pipeline/src/analysis/run_analysis.py --analysis basic",
        "Testing basic xG analysis"
    )
    
    # Test business analysis
    success2 = run_command_safely(
        "python ../data_pipeline/src/analysis/run_analysis.py --analysis business", 
        "Testing business constraint analysis"
    )
    
    # Test quick runner
    success3 = run_command_safely(
        "python run_nhl_analysis.py",
        "Testing quick analysis runner"
    )
    
    return all([success1, success2, success3])

def generate_presentation_visuals():
    """Generate all visualizations needed for presentation."""
    print_section("GENERATING PRESENTATION VISUALS")
    
    # Create report-images directory if it doesn't exist
    os.makedirs("report-images", exist_ok=True)
    
    # Generate comprehensive visualizations
    success = run_command_safely(
        "python ../data_pipeline/src/visualization/report_visualization_package.py",
        "Generating professional visualizations"
    )
    
    if success:
        # List generated files
        report_dir = Path("report-images")
        if report_dir.exists():
            files = list(report_dir.glob("*.png"))
            print(f"\n📸 Generated {len(files)} visualization files:")
            for file in sorted(files):
                print(f"   - {file.name}")
        
        # Check for key visualizations
        key_visuals = [
            "01_ice_rink_heatmap.png",
            "04_model_evolution.png", 
            "06_business_impact_dashboard.png",
            "07_technical_architecture.png"
        ]
        
        missing = []
        for visual in key_visuals:
            if not (report_dir / visual).exists():
                missing.append(visual)
        
        if missing:
            print(f"\n⚠️  Missing key visuals: {missing}")
        else:
            print(f"\n✅ All key presentation visuals generated!")
    
    return success

def test_demo_scenarios():
    """Test specific demo scenarios for the presentation."""
    print_section("DEMO SCENARIO TESTING")
    
    scenarios = [
        {
            "name": "Data Verification",
            "command": "python ../data_pipeline/src/data/verify_data.py",
            "description": "Verify data quality and statistics"
        },
        {
            "name": "Model Training Demo", 
            "command": "cd ../data_pipeline/src && python -c \"from models.nhl_xg_core import NHLxGAnalyzer; import os; os.chdir('../../project1_supervised'); a=NHLxGAnalyzer(); a.load_shot_data(); print('✅ Data loaded successfully')\"",
            "description": "Quick model loading test"
        },
        {
            "name": "Feature Engineering Demo",
            "command": "cd ../data_pipeline/src && python -c \"from models.nhl_xg_core import NHLxGAnalyzer; import os; os.chdir('../../project1_supervised'); a=NHLxGAnalyzer(); a.load_shot_data(); a.engineer_features(); print(f'✅ Engineered features for {len(a.shot_events):,} shots')\"",
            "description": "Feature engineering demonstration"
        }
    ]
    
    results = []
    for scenario in scenarios:
        success = run_command_safely(
            scenario["command"],
            scenario["description"]
        )
        results.append(success)
        time.sleep(1)  # Brief pause between tests
    
    return all(results)

def create_demo_data_samples():
    """Create sample data for demo purposes."""
    print_section("CREATING DEMO DATA SAMPLES")
    
    try:
        import sqlite3
        import pandas as pd
        import json
        
        conn = sqlite3.connect("nhl_stats.db")
        
        # Get sample interesting shots for demo
        sample_query = """
        SELECT 
            e.gamePk,
            e.eventType,
            e.x,
            e.y,
            e.details,
            g.gameDate
        FROM events e
        JOIN games g ON e.gamePk = g.gamePk
        WHERE e.eventType IN ('goal', 'shot-on-goal')
        AND e.x IS NOT NULL 
        AND e.y IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 10
        """
        
        sample_shots = pd.read_sql_query(sample_query, conn)
        
        # Create demo scenarios
        demo_scenarios = []
        for _, shot in sample_shots.iterrows():
            try:
                details = json.loads(shot['details'])
                x, y = shot['x'], shot['y']
                
                # Calculate basic features for demo
                distance = min(
                    ((x - 89)**2 + y**2)**0.5,
                    ((x + 89)**2 + y**2)**0.5
                )
                
                scenario = {
                    'game': shot['gamePk'],
                    'type': shot['eventType'],
                    'coordinates': f"({x:.1f}, {y:.1f})",
                    'distance': f"{distance:.1f} feet",
                    'date': shot['gameDate']
                }
                demo_scenarios.append(scenario)
            except:
                continue
        
        conn.close()
        
        # Save demo scenarios
        with open("demo_scenarios.json", "w") as f:
            json.dump(demo_scenarios, f, indent=2)
        
        print(f"✅ Created {len(demo_scenarios)} demo scenarios")
        print("📁 Saved to: demo_scenarios.json")
        
        # Show sample scenarios
        print("\n🎯 Sample demo scenarios:")
        for i, scenario in enumerate(demo_scenarios[:3]):
            print(f"   {i+1}. {scenario['type']} from {scenario['distance']} at {scenario['coordinates']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating demo data: {e}")
        return False

def check_presentation_readiness():
    """Final check of presentation readiness."""
    print_section("PRESENTATION READINESS CHECK")
    
    checklist = [
        ("NHL Database", os.path.exists("nhl_stats.db")),
        ("Source Code", os.path.exists("src")),
        ("Visualizations", os.path.exists("report-images")),
        ("Demo Runner", os.path.exists("run_nhl_analysis.py")),
        ("Presentation Script", os.path.exists("presentation_script.md")),
        ("Demo Scenarios", os.path.exists("demo_scenarios.json"))
    ]
    
    print("📋 Presentation Checklist:")
    all_ready = True
    for item, status in checklist:
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {item}")
        if not status:
            all_ready = False
    
    if all_ready:
        print(f"\n🎉 PRESENTATION READY!")
        print(f"📝 Script: presentation_script.md")
        print(f"🖼️  Visuals: report-images/")
        print(f"🎯 Demo data: demo_scenarios.json")
        print(f"\n💡 Recording tips:")
        print(f"   - Practice the demo commands first")
        print(f"   - Use 1920x1080 screen resolution")
        print(f"   - Test audio quality")
        print(f"   - Have backup screenshots ready")
    else:
        print(f"\n⚠️  Some components missing - check above")
    
    return all_ready

def main():
    """Main preparation workflow."""
    print("🎬 NHL xG PRESENTATION DEMO PREPARATION")
    print("=" * 60)
    print("This script will prepare and test all components for your presentation")
    
    # Step 1: Check data
    if not check_data_availability():
        print("\n💡 Please run data collection first:")
        print("   python src/data/scrape_nhl_data.py")
        return
    
    # Step 2: Test core functionality
    if not test_core_functionality():
        print("\n⚠️  Core functionality issues detected")
        print("   Check error messages above")
    
    # Step 3: Generate visuals
    if not generate_presentation_visuals():
        print("\n⚠️  Visualization generation issues")
        print("   Check error messages above")
    
    # Step 4: Test demo scenarios
    if not test_demo_scenarios():
        print("\n⚠️  Demo scenario issues")
        print("   Check error messages above")
    
    # Step 5: Create demo data
    create_demo_data_samples()
    
    # Step 6: Final readiness check
    check_presentation_readiness()
    
    print(f"\n🎬 PREPARATION COMPLETE!")
    print(f"📖 Next steps:")
    print(f"   1. Review presentation_script.md")
    print(f"   2. Practice the demo commands")
    print(f"   3. Create presentation slides")
    print(f"   4. Record your video!")

if __name__ == "__main__":
    main() 